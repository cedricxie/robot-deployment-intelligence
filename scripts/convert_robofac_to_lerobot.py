#!/usr/bin/env python3
"""Convert a RoboFAC simulation subset into LeRobot dataset format (v2.1, optionally v3.0).

RoboFAC (MINT-SJTU/RoboFAC-dataset) stores per-leaf MP4s + ToolsTask_traj.{json,h5}.
This script selects a diverse preview of episodes, writes a LeRobot v2.1 layout that
the Hub visualizer accepts, and can optionally run the official v2.1→v3.0 converter.

Example:
  source /workspace/.venv-lerobot/bin/activate
  python scripts/convert_robofac_to_lerobot.py \\
      --raw-dir data/raw/robofac \\
      --out-dir data/processed/robofac_lerobot_preview \\
      --max-episodes 32 \\
      --to-v30
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import h5py
except ImportError as e:  # pragma: no cover
    raise SystemExit("h5py is required: pip install h5py") from e

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:  # pragma: no cover
    raise SystemExit("pandas/pyarrow required: pip install pandas pyarrow") from e

CAMERA_KEY = "observation.images.main"
ACTION_DIM = 8
JOINT_NAMES = [f"joint_{i}" for i in range(ACTION_DIM)]
DEFAULT_FPS = 30
CHUNKS_SIZE = 1000


@dataclass
class EpisodeCandidate:
    rel_video: str
    abs_video: Path
    leaf_dir: Path
    task_folder: str
    scenario: str
    unique_id: str
    episode_id_in_leaf: int
    elapsed_steps: int
    success: bool
    category: str  # success | failure_forget | failure_error | failure | mixed
    task_description: str


def _category_for(rel: str, success: bool) -> str:
    low = rel.lower()
    if "forget" in low:
        return "failure_forget"
    if "gripper_error" in low or "/stack_error/" in low and "success_data" not in low:
        return "failure_error"
    if success or "success_data" in low:
        return "success" if success else "failure"
    return "failure" if not success else "success"


def load_qa_task_map(qa_path: Path | None) -> dict[str, str]:
    """Map video relative path -> natural-language task description from training_qa.json."""
    if qa_path is None or not qa_path.exists():
        return {}
    data = json.loads(qa_path.read_text())
    out: dict[str, str] = {}
    task_q_keywords = ("what task", "task is the robot", "carrying out", "engaged in", "performing")
    for item in data:
        video = item.get("video")
        if not video or video in out:
            continue
        convs = item.get("conversations") or []
        for i, turn in enumerate(convs):
            if turn.get("from") != "human":
                continue
            q = (turn.get("value") or "").lower()
            if any(k in q for k in task_q_keywords):
                if i + 1 < len(convs) and convs[i + 1].get("from") == "assistant":
                    out[video] = convs[i + 1].get("value") or ""
                    break
    return out


def load_qa_success_map(qa_path: Path | None) -> dict[str, bool | None]:
    """Map video -> success from QA answers when present."""
    if qa_path is None or not qa_path.exists():
        return {}
    data = json.loads(qa_path.read_text())
    out: dict[str, bool | None] = {}
    for item in data:
        video = item.get("video")
        if not video:
            continue
        convs = item.get("conversations") or []
        for i, turn in enumerate(convs):
            if turn.get("from") != "human":
                continue
            q = (turn.get("value") or "").lower()
            if "success" not in q and "successfully" not in q:
                continue
            if i + 1 >= len(convs):
                continue
            ans = (convs[i + 1].get("value") or "").strip().lower()
            if ans.startswith("yes"):
                out[video] = True
            elif ans.startswith("no"):
                out[video] = False
    return out


def iter_candidates(raw_dir: Path, qa_tasks: dict[str, str]) -> list[EpisodeCandidate]:
    sim = raw_dir / "simulation_data"
    if not sim.is_dir():
        raise FileNotFoundError(f"Missing simulation_data under {raw_dir}")

    cands: list[EpisodeCandidate] = []
    for traj_json in sorted(sim.rglob("ToolsTask_traj.json")):
        leaf = traj_json.parent
        try:
            meta = json.loads(traj_json.read_text())
        except json.JSONDecodeError:
            continue
        episodes = meta.get("episodes") or []
        # task folder: first path component under simulation_data (may be success_data/...)
        rel_leaf = leaf.relative_to(sim)
        parts = rel_leaf.parts
        if parts[0] == "success_data" and len(parts) >= 2:
            task_folder = f"success_data/{parts[1]}"
            scenario = parts[2] if len(parts) >= 3 else parts[-1]
        else:
            task_folder = parts[0]
            scenario = parts[1] if len(parts) >= 2 else parts[-1]

        for ep in episodes:
            uid = ep.get("unique_id")
            if not uid:
                continue
            mp4 = leaf / f"{uid}.mp4"
            if not mp4.is_file():
                continue
            rel_video = str((rel_leaf / f"{uid}.mp4").as_posix())
            success = bool(ep.get("success", False))
            task_desc = qa_tasks.get(rel_video) or qa_tasks.get(
                str(Path(task_folder.split("/")[-1]) / scenario / f"{uid}.mp4")
            )
            if not task_desc:
                # Fallback: humanize task folder
                base = task_folder.split("/")[-1]
                task_desc = base.replace("-", " ")
            cands.append(
                EpisodeCandidate(
                    rel_video=rel_video,
                    abs_video=mp4,
                    leaf_dir=leaf,
                    task_folder=task_folder,
                    scenario=scenario,
                    unique_id=uid,
                    episode_id_in_leaf=int(ep.get("episode_id", 0)),
                    elapsed_steps=int(ep.get("elapsed_steps", 0)),
                    success=success,
                    category=_category_for(rel_video, success),
                    task_description=task_desc,
                )
            )
    return cands


def select_diverse(
    cands: list[EpisodeCandidate],
    max_episodes: int,
    seed: int,
) -> list[EpisodeCandidate]:
    """Pick a diverse mix across tasks and success/failure categories."""
    rng = random.Random(seed)
    by_cat: dict[str, list[EpisodeCandidate]] = defaultdict(list)
    for c in cands:
        by_cat[c.category].append(c)

    # Target mix: ~40% success, ~30% forget/error, ~30% other failure
    quotas = {
        "success": max(1, int(max_episodes * 0.40)),
        "failure_forget": max(1, int(max_episodes * 0.15)),
        "failure_error": max(1, int(max_episodes * 0.15)),
        "failure": max(1, int(max_episodes * 0.30)),
    }
    # Normalize quotas to max_episodes
    total_q = sum(quotas.values())
    while total_q > max_episodes:
        for k in sorted(quotas, key=lambda x: -quotas[x]):
            if quotas[k] > 1 and total_q > max_episodes:
                quotas[k] -= 1
                total_q -= 1

    selected: list[EpisodeCandidate] = []
    used_uids: set[str] = set()
    used_tasks: dict[str, int] = defaultdict(int)

    def pick_from(pool: list[EpisodeCandidate], n: int) -> list[EpisodeCandidate]:
        pool = [c for c in pool if c.unique_id not in used_uids]
        # Prefer under-represented task folders
        pool.sort(key=lambda c: (used_tasks[c.task_folder], rng.random()))
        out = []
        for c in pool:
            if len(out) >= n:
                break
            out.append(c)
            used_uids.add(c.unique_id)
            used_tasks[c.task_folder] += 1
        return out

    for cat, n in quotas.items():
        selected.extend(pick_from(by_cat.get(cat, []), n))

    # Fill remaining from all
    if len(selected) < max_episodes:
        rest = [c for c in cands if c.unique_id not in used_uids]
        selected.extend(pick_from(rest, max_episodes - len(selected)))

    rng.shuffle(selected)
    return selected[:max_episodes]


def load_actions(leaf_dir: Path, episode_id: int) -> np.ndarray:
    h5_path = leaf_dir / "ToolsTask_traj.h5"
    with h5py.File(h5_path, "r") as f:
        key = f"traj_{episode_id}"
        if key not in f:
            raise KeyError(f"{key} missing in {h5_path}")
        actions = np.asarray(f[key]["actions"], dtype=np.float32)
        success_arr = np.asarray(f[key]["success"], dtype=bool)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        # Pad / truncate to ACTION_DIM
        out = np.zeros((actions.shape[0], ACTION_DIM), dtype=np.float32)
        dim = min(ACTION_DIM, actions.shape[1] if actions.ndim == 2 else 0)
        if actions.ndim == 2:
            out[:, :dim] = actions[:, :dim]
        actions = out
    return actions, success_arr


def probe_video(path: Path) -> dict[str, Any]:
    """Return width, height, fps, n_frames via ffprobe (preferred) or pyav."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ]
        raw = subprocess.check_output(cmd, text=True)
        stream = json.loads(raw)["streams"][0]
        num, den = stream["r_frame_rate"].split("/")
        fps = float(num) / float(den) if float(den) else DEFAULT_FPS
        n_frames = int(stream["nb_frames"]) if stream.get("nb_frames") not in (None, "N/A") else None
        if n_frames is None and stream.get("duration"):
            n_frames = int(round(float(stream["duration"]) * fps))
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": fps,
            "n_frames": n_frames or 0,
            "codec": stream.get("codec_name", "h264"),
        }
    except Exception:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else DEFAULT_FPS
            n = stream.frames or 0
            if not n:
                n = sum(1 for _ in container.decode(stream))
            return {
                "width": int(stream.width),
                "height": int(stream.height),
                "fps": fps,
                "n_frames": int(n),
                "codec": stream.codec_context.name if stream.codec_context else "h264",
            }


def numeric_stats(arr: np.ndarray) -> dict[str, Any]:
    """Per-feature stats in LeRobot v2.1 list form."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    # images handled separately
    amin = arr.min(axis=0)
    amax = arr.max(axis=0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return {
        "min": amin.astype(float).tolist(),
        "max": amax.astype(float).tolist(),
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "count": [int(arr.shape[0])],
    }


def image_placeholder_stats(n: int) -> dict[str, Any]:
    """Cheap per-channel image stats (visualizer / training-normalization friendly)."""
    # Shape convention in stats: [[[c]]] for channel-wise over spatial dims
    mean = [[[0.5]], [[0.5]], [[0.5]]]
    std = [[[0.25]], [[0.25]], [[0.25]]]
    return {
        "min": [[[0.0]], [[0.0]], [[0.0]]],
        "max": [[[1.0]], [[1.0]], [[1.0]]],
        "mean": mean,
        "std": std,
        "count": [int(n)],
    }


def write_episode_parquet(
    path: Path,
    *,
    episode_index: int,
    task_index: int,
    global_index_start: int,
    actions: np.ndarray,
    success_per_frame: np.ndarray,
    fps: float,
) -> int:
    n = int(actions.shape[0])
    # observation.state: use action targets as proprio proxy (RoboFAC obs_mode=none)
    states = actions.copy()
    frame_index = np.arange(n, dtype=np.int64)
    timestamps = (frame_index.astype(np.float32) / float(fps))
    episode_index_col = np.full(n, episode_index, dtype=np.int64)
    index = np.arange(global_index_start, global_index_start + n, dtype=np.int64)
    task_index_col = np.full(n, task_index, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    done[-1] = True
    # next.success: episode-level label broadcast
    ep_success = bool(success_per_frame[-1]) if len(success_per_frame) else False
    next_success = np.full(n, ep_success, dtype=bool)

    table = pa.table(
        {
            "observation.state": [states[i].astype(np.float32).tolist() for i in range(n)],
            "action": [actions[i].astype(np.float32).tolist() for i in range(n)],
            "timestamp": timestamps.tolist(),
            "frame_index": frame_index.tolist(),
            "episode_index": episode_index_col.tolist(),
            "index": index.tolist(),
            "task_index": task_index_col.tolist(),
            "next.done": done.tolist(),
            "next.success": next_success.tolist(),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return n


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_v21(
    selected: list[EpisodeCandidate],
    out_dir: Path,
    fps_override: float | None = None,
) -> dict[str, Any]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "meta").mkdir(parents=True)
    (out_dir / "data" / "chunk-000").mkdir(parents=True)
    (out_dir / "videos" / "chunk-000" / CAMERA_KEY).mkdir(parents=True)

    # Probe first video for resolution / fps
    probe0 = probe_video(selected[0].abs_video)
    fps = float(fps_override or probe0["fps"] or DEFAULT_FPS)
    height, width = int(probe0["height"]), int(probe0["width"])

    tasks: dict[str, int] = {}
    episodes_meta: list[dict[str, Any]] = []
    episodes_stats: list[dict[str, Any]] = []
    total_frames = 0
    global_index = 0

    for ep_idx, cand in enumerate(selected):
        actions, success_arr = load_actions(cand.leaf_dir, cand.episode_id_in_leaf)
        probe = probe_video(cand.abs_video)
        n_vid = probe["n_frames"] or actions.shape[0]
        n = min(int(actions.shape[0]), int(n_vid)) if n_vid else int(actions.shape[0])
        actions = actions[:n]
        success_arr = success_arr[:n] if len(success_arr) >= n else np.full(n, cand.success)

        outcome = "success" if cand.success else "failure"
        task_str = f"{cand.task_description} [{outcome}]"
        if task_str not in tasks:
            tasks[task_str] = len(tasks)
        task_index = tasks[task_str]

        ep_name = f"episode_{ep_idx:06d}"
        data_path = out_dir / "data" / "chunk-000" / f"{ep_name}.parquet"
        video_dst = out_dir / "videos" / "chunk-000" / CAMERA_KEY / f"{ep_name}.mp4"
        write_episode_parquet(
            data_path,
            episode_index=ep_idx,
            task_index=task_index,
            global_index_start=global_index,
            actions=actions,
            success_per_frame=success_arr,
            fps=fps,
        )
        link_or_copy(cand.abs_video, video_dst)

        episodes_meta.append(
            {
                "episode_index": ep_idx,
                "tasks": [task_str],
                "length": n,
                # Extra provenance (ignored by visualizer, useful for us)
                "robofac_video": cand.rel_video,
                "robofac_unique_id": cand.unique_id,
                "robofac_task_folder": cand.task_folder,
                "robofac_scenario": cand.scenario,
                "robofac_success": cand.success,
                "robofac_category": cand.category,
            }
        )

        # Stats
        ts = np.arange(n, dtype=np.float32) / fps
        fi = np.arange(n, dtype=np.float32)
        ep_stats = {
            "episode_index": ep_idx,
            "stats": {
                "observation.state": numeric_stats(actions),
                "action": numeric_stats(actions),
                "timestamp": numeric_stats(ts),
                "frame_index": numeric_stats(fi),
                "episode_index": numeric_stats(np.full(n, ep_idx, dtype=np.float32)),
                "index": numeric_stats(np.arange(global_index, global_index + n, dtype=np.float32)),
                "task_index": numeric_stats(np.full(n, task_index, dtype=np.float32)),
                CAMERA_KEY: image_placeholder_stats(n),
            },
        }
        episodes_stats.append(ep_stats)

        global_index += n
        total_frames += n
        print(
            f"[{ep_idx+1}/{len(selected)}] {cand.rel_video} "
            f"frames={n} success={cand.success} cat={cand.category}"
        )

    # tasks.jsonl
    with (out_dir / "meta" / "tasks.jsonl").open("w") as f:
        for task, idx in sorted(tasks.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    with (out_dir / "meta" / "episodes.jsonl").open("w") as f:
        for row in episodes_meta:
            f.write(json.dumps(row) + "\n")

    with (out_dir / "meta" / "episodes_stats.jsonl").open("w") as f:
        for row in episodes_stats:
            f.write(json.dumps(row) + "\n")

    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "total_episodes": len(selected),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "chunks_size": CHUNKS_SIZE,
        "fps": int(round(fps)),
        "splits": {"train": f"0:{len(selected)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            CAMERA_KEY: {
                "dtype": "video",
                "shape": [height, width, 3],
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": float(fps),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "has_audio": False,
                },
            },
            "observation.state": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": {"motors": JOINT_NAMES},
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": {"motors": JOINT_NAMES},
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "next.done": {"dtype": "bool", "shape": [1], "names": None},
            "next.success": {"dtype": "bool", "shape": [1], "names": None},
        },
    }
    (out_dir / "meta" / "info.json").write_text(json.dumps(info, indent=2) + "\n")

    write_dataset_card(out_dir, "local/robofac_preview", info)

    # Manifest for reproducibility
    manifest = {
        "n_episodes": len(selected),
        "total_frames": total_frames,
        "fps": fps,
        "episodes": [
            {
                "episode_index": i,
                "rel_video": c.rel_video,
                "success": c.success,
                "category": c.category,
                "task": episodes_meta[i]["tasks"][0],
                "length": episodes_meta[i]["length"],
            }
            for i, c in enumerate(selected)
        ],
    }
    (out_dir / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return info


def write_dataset_card(out_dir: Path, repo_id: str, info: dict[str, Any]) -> None:
    readme = f"""---
license: apache-2.0
task_categories:
  - robotics
tags:
  - LeRobot
  - LeRobotDataset
  - RoboFAC
  - panda
  - simulation
---

# RoboFAC → LeRobot preview (`{repo_id}`)

Small preview subset of [MINT-SJTU/RoboFAC-dataset](https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset)
converted to **LeRobot {info.get("codebase_version", "v2.1+")}** for the
[dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset).

| | |
|--|--|
| Episodes | **{info["total_episodes"]}** |
| Frames | **{info["total_frames"]}** |
| FPS | **{info["fps"]}** |
| Robot | `{info.get("robot_type")}` |
| Camera | `{CAMERA_KEY}` |
| Format | `{info.get("codebase_version")}` |

## Notes

- Actions / `observation.state` are 8-D joint targets from RoboFAC `ToolsTask_traj.h5` (`pd_joint_pos`).
  True proprioception was not stored (`obs_mode=none`).
- Task strings include `[success]` / `[failure]` from trajectory labels.
- Visualization / format preview only — not a full RoboFAC port.

## Visualize

`https://huggingface.co/spaces/lerobot/visualize_dataset/{repo_id}`

## Reproduce

```bash
python scripts/convert_robofac_to_lerobot.py \
  --raw-dir data/raw/robofac \
  --out-dir data/processed/robofac_lerobot_preview \
  --max-episodes 32 --to-v30
```
"""
    (out_dir / "README.md").write_text(readme)


def convert_to_v30(out_dir: Path, repo_id: str) -> None:
    """Run official LeRobot v2.1 → v3.0 converter in-place."""
    # Preserve extras the converter may drop while shuffling directories.
    manifest_src = out_dir / "conversion_manifest.json"
    manifest_backup = None
    if manifest_src.exists():
        manifest_backup = manifest_src.read_text()

    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.convert_dataset_v21_to_v30",
        f"--repo-id={repo_id}",
        f"--root={out_dir}",
        "--push-to-hub=false",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

    # Official converter may leave <root>_old; remove to save disk.
    old = Path(str(out_dir) + "_old")
    if old.exists():
        shutil.rmtree(old)

    if manifest_backup is not None:
        (out_dir / "conversion_manifest.json").write_text(manifest_backup)

    info = json.loads((out_dir / "meta" / "info.json").read_text())
    write_dataset_card(out_dir, repo_id, info)


def structural_validate(out_dir: Path) -> None:
    info = json.loads((out_dir / "meta" / "info.json").read_text())
    version = info["codebase_version"]
    assert info["total_episodes"] > 0
    assert info["total_frames"] > 0
    if version == "v2.1":
        assert (out_dir / "meta" / "episodes.jsonl").exists()
        assert (out_dir / "meta" / "tasks.jsonl").exists()
        assert (out_dir / "meta" / "episodes_stats.jsonl").exists()
        n = info["total_episodes"]
        for i in range(n):
            p = out_dir / "data" / "chunk-000" / f"episode_{i:06d}.parquet"
            v = out_dir / "videos" / "chunk-000" / CAMERA_KEY / f"episode_{i:06d}.mp4"
            assert p.is_file(), p
            assert v.is_file(), v
            t = pq.read_table(p)
            assert t.num_rows > 0
    elif version == "v3.0":
        assert (out_dir / "meta" / "episodes").exists()
        assert list((out_dir / "data").rglob("*.parquet"))
        assert list((out_dir / "videos").rglob("*.mp4"))
    else:
        raise AssertionError(f"Unexpected version {version}")
    print(f"Structural validation OK ({version}): "
          f"{info['total_episodes']} episodes, {info['total_frames']} frames")


def try_lerobot_load(out_dir: Path, repo_id: str) -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("LeRobot not installed; skip LeRobotDataset load check")
        return
    info = json.loads((out_dir / "meta" / "info.json").read_text())
    if info["codebase_version"] != "v3.0":
        print(f"codebase_version={info['codebase_version']}; "
              "LeRobot>=0.4 expects v3.0 — skip load (convert with --to-v30)")
        return
    ds = LeRobotDataset(repo_id=repo_id, root=out_dir, download_videos=False)
    print(f"LeRobotDataset load OK: episodes={ds.num_episodes} frames={ds.num_frames}")
    _ = ds[0]
    print("frame0 keys:", sorted(ds[0].keys()))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/robofac"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/robofac_lerobot_preview"))
    p.add_argument("--qa-json", type=Path, default=None, help="Defaults to <raw-dir>/training_qa.json")
    p.add_argument("--max-episodes", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=float, default=None, help="Override FPS (default: probe from video)")
    p.add_argument("--repo-id", type=str, default="cedricxie/robofac-lerobot-preview")
    p.add_argument("--to-v30", action="store_true", help="Convert v2.1 output to v3.0 in-place")
    p.add_argument("--skip-validate", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    out_dir = args.out_dir.resolve()
    qa_path = (args.qa_json or (raw_dir / "training_qa.json")).resolve()

    print(f"Scanning candidates under {raw_dir} ...")
    qa_tasks = load_qa_task_map(qa_path)
    print(f"QA task descriptions: {len(qa_tasks)}")
    cands = iter_candidates(raw_dir, qa_tasks)
    print(f"Candidates with video+traj: {len(cands)}")
    by_cat = defaultdict(int)
    for c in cands:
        by_cat[c.category] += 1
    print("Category counts:", dict(by_cat))

    selected = select_diverse(cands, args.max_episodes, args.seed)
    print(f"Selected {len(selected)} episodes")

    info = build_v21(selected, out_dir, fps_override=args.fps)
    if not args.skip_validate:
        structural_validate(out_dir)

    if args.to_v30:
        convert_to_v30(out_dir, args.repo_id)
        if not args.skip_validate:
            structural_validate(out_dir)
            try_lerobot_load(out_dir, args.repo_id)

    print(f"Done. Dataset at {out_dir}")
    print(f"Visualizer URL (after Hub upload): "
          f"https://huggingface.co/spaces/lerobot/visualize_dataset?path={args.repo_id}")


if __name__ == "__main__":
    main()
