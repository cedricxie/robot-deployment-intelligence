#!/usr/bin/env python3
"""Convert a RoboFAC sim (+ optional realworld) subset into LeRobot format.

RoboFAC (MINT-SJTU/RoboFAC-dataset) stores:
  - simulation_data: per-leaf MP4s + ToolsTask_traj.{json,h5}
  - realworld_data/so100_*: video-only trees (often AV1; no state/action)

This script selects a diverse preview, writes LeRobot v2.1 (visualizer-friendly),
and can optionally run the official v2.1→v3.0 converter.

Realworld episodes are remapped onto the same schema as sim:
  observation.images.above → observation.images.main, 8-D zero action/state,
  task tags like "realworld: so100_stack_cube_error [failure]", next.success=False
  for *_error tasks. Non-H.264 clips are re-encoded for Hub/browser playback.

Example:
  source /workspace/.venv-lerobot/bin/activate
  python scripts/convert_robofac_to_lerobot.py \\
      --raw-dir data/raw/robofac \\
      --out-dir data/processed/robofac_lerobot_preview \\
      --max-episodes 32 \\
      --include-realworld --max-realworld-episodes 18 \\
      --to-v30
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
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
    import pandas as pd  # noqa: F401
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:  # pragma: no cover
    raise SystemExit("pandas/pyarrow required: pip install pandas pyarrow") from e

CAMERA_KEY = "observation.images.main"
REALWORLD_CAMERA = "observation.images.above"  # remap → CAMERA_KEY
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
    category: str  # success | failure_forget | failure_error | failure | realworld_error
    task_description: str
    source: str = "sim"  # sim | realworld
    needs_h264: bool = False


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


def load_realworld_task_map(raw_dir: Path) -> dict[str, str]:
    """Map realworld video rel path -> task description from test_qa_realworld annos."""
    qa_dir = raw_dir / "test_qa_realworld"
    out: dict[str, str] = {}
    if not qa_dir.is_dir():
        return out
    for path in sorted(qa_dir.glob("annos_per_video_split*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        for _uid, item in data.items():
            if not isinstance(item, dict):
                continue
            video = item.get("video")
            if not video or video in out:
                continue
            # Prefer Task identification assistant answer; else item["task"]
            annos = item.get("annos") or {}
            desc = ""
            for turn_pair in annos.get("Task identification") or []:
                if turn_pair.get("from") == "assistant":
                    desc = (turn_pair.get("value") or "").strip()
                    break
            if not desc:
                desc = (item.get("task") or "").strip()
            if desc:
                out[video] = desc
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
                    source="sim",
                    needs_h264=False,
                )
            )
    return cands


def iter_realworld_candidates(
    raw_dir: Path,
    rw_tasks: dict[str, str],
) -> list[EpisodeCandidate]:
    """Scan realworld_data/so100_*/videos/.../observation.images.above/*.mp4."""
    rw_root = raw_dir / "realworld_data"
    if not rw_root.is_dir():
        raise FileNotFoundError(f"Missing realworld_data under {raw_dir}")

    cands: list[EpisodeCandidate] = []
    for task_dir in sorted(rw_root.glob("so100_*")):
        if not task_dir.is_dir():
            continue
        task_folder = task_dir.name
        cam_dir = task_dir / "videos" / "chunk-000" / REALWORLD_CAMERA
        if not cam_dir.is_dir():
            continue
        is_error = task_folder.endswith("_error")
        for mp4 in sorted(cam_dir.glob("episode_*.mp4")):
            m = re.match(r"episode_(\d+)\.mp4$", mp4.name)
            if not m:
                continue
            ep_id = int(m.group(1))
            rel_video = str(
                (Path(task_folder) / "videos" / "chunk-000" / REALWORLD_CAMERA / mp4.name).as_posix()
            )
            # Humanized fallback: so100_stack_cube_error → stack cube
            human = task_folder
            if human.startswith("so100_"):
                human = human[len("so100_") :]
            if human.endswith("_error"):
                human = human[: -len("_error")]
            human = human.replace("_", " ")
            task_desc = rw_tasks.get(rel_video) or human
            # Tagged description used later; keep base NL here
            success = False if is_error else False  # all current trees are *_error
            cands.append(
                EpisodeCandidate(
                    rel_video=rel_video,
                    abs_video=mp4,
                    leaf_dir=cam_dir,
                    task_folder=task_folder,
                    scenario="realworld",
                    unique_id=f"{task_folder}/{mp4.stem}",
                    episode_id_in_leaf=ep_id,
                    elapsed_steps=0,
                    success=success,
                    category="realworld_error" if is_error else "realworld",
                    task_description=task_desc,
                    source="realworld",
                    needs_h264=True,  # RoboFAC realworld ships AV1
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

    quotas = {
        "success": max(1, int(max_episodes * 0.40)),
        "failure_forget": max(1, int(max_episodes * 0.15)),
        "failure_error": max(1, int(max_episodes * 0.15)),
        "failure": max(1, int(max_episodes * 0.30)),
    }
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

    if len(selected) < max_episodes:
        rest = [c for c in cands if c.unique_id not in used_uids]
        selected.extend(pick_from(rest, max_episodes - len(selected)))

    rng.shuffle(selected)
    return selected[:max_episodes]


def select_realworld_diverse(
    cands: list[EpisodeCandidate],
    max_episodes: int,
    seed: int,
) -> list[EpisodeCandidate]:
    """Evenly sample across so100_* task folders."""
    rng = random.Random(seed + 7)
    by_task: dict[str, list[EpisodeCandidate]] = defaultdict(list)
    for c in cands:
        by_task[c.task_folder].append(c)
    for pool in by_task.values():
        rng.shuffle(pool)

    tasks = sorted(by_task.keys())
    if not tasks or max_episodes <= 0:
        return []

    # Round-robin so each task gets ~floor(N/T) or ceil(N/T)
    selected: list[EpisodeCandidate] = []
    idx = {t: 0 for t in tasks}
    while len(selected) < max_episodes:
        progressed = False
        for t in tasks:
            if len(selected) >= max_episodes:
                break
            i = idx[t]
            if i < len(by_task[t]):
                selected.append(by_task[t][i])
                idx[t] = i + 1
                progressed = True
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def load_actions(leaf_dir: Path, episode_id: int) -> tuple[np.ndarray, np.ndarray]:
    h5_path = leaf_dir / "ToolsTask_traj.h5"
    with h5py.File(h5_path, "r") as f:
        key = f"traj_{episode_id}"
        if key not in f:
            raise KeyError(f"{key} missing in {h5_path}")
        actions = np.asarray(f[key]["actions"], dtype=np.float32)
        success_arr = np.asarray(f[key]["success"], dtype=bool)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
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
    # observation.state: use action targets as proprio proxy (RoboFAC obs_mode=none);
    # realworld uses zero placeholders matching ACTION_DIM.
    states = actions.copy()
    frame_index = np.arange(n, dtype=np.int64)
    timestamps = frame_index.astype(np.float32) / float(fps)
    episode_index_col = np.full(n, episode_index, dtype=np.int64)
    index = np.arange(global_index_start, global_index_start + n, dtype=np.int64)
    task_index_col = np.full(n, task_index, dtype=np.int64)
    done = np.zeros(n, dtype=bool)
    done[-1] = True
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


def ensure_h264(src: Path, dst: Path, *, force: bool = False) -> None:
    """Hardlink/copy if already H.264; otherwise re-encode with libx264."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    codec = ""
    if not force:
        try:
            codec = (probe_video(src).get("codec") or "").lower()
        except Exception:
            codec = ""
    if codec in ("h264", "avc1", "avc") and not force:
        link_or_copy(src, dst)
        return
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def task_string_for(cand: EpisodeCandidate) -> str:
    outcome = "success" if cand.success else "failure"
    if cand.source == "realworld":
        # Clear domain tag + folder name for filtering in the visualizer
        return f"realworld: {cand.task_folder} [{outcome}]"
    return f"{cand.task_description} [{outcome}]"


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

    probe0 = probe_video(selected[0].abs_video)
    fps = float(fps_override or probe0["fps"] or DEFAULT_FPS)
    # Feature shape from first (typically sim) episode; realworld may differ in
    # native resolution — Hub visualizer plays MP4s directly; LeRobot video
    # dtype decodes per-file. Preview-only, not for batched mixed-res training.
    height, width = int(probe0["height"]), int(probe0["width"])

    tasks: dict[str, int] = {}
    episodes_meta: list[dict[str, Any]] = []
    episodes_stats: list[dict[str, Any]] = []
    total_frames = 0
    global_index = 0
    n_sim = sum(1 for c in selected if c.source == "sim")
    n_real = sum(1 for c in selected if c.source == "realworld")

    for ep_idx, cand in enumerate(selected):
        probe = probe_video(cand.abs_video)
        if cand.source == "realworld":
            n = int(probe["n_frames"] or 0)
            if n <= 0:
                raise RuntimeError(f"Could not determine frame count for {cand.abs_video}")
            actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
            success_arr = np.full(n, cand.success, dtype=bool)
        else:
            actions, success_arr = load_actions(cand.leaf_dir, cand.episode_id_in_leaf)
            n_vid = probe["n_frames"] or actions.shape[0]
            n = min(int(actions.shape[0]), int(n_vid)) if n_vid else int(actions.shape[0])
            actions = actions[:n]
            success_arr = success_arr[:n] if len(success_arr) >= n else np.full(n, cand.success)

        task_str = task_string_for(cand)
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
        if cand.needs_h264:
            ensure_h264(cand.abs_video, video_dst, force=True)
        else:
            link_or_copy(cand.abs_video, video_dst)

        episodes_meta.append(
            {
                "episode_index": ep_idx,
                "tasks": [task_str],
                "length": n,
                "robofac_video": cand.rel_video,
                "robofac_unique_id": cand.unique_id,
                "robofac_task_folder": cand.task_folder,
                "robofac_scenario": cand.scenario,
                "robofac_success": cand.success,
                "robofac_category": cand.category,
                "robofac_source": cand.source,
            }
        )

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
            f"[{ep_idx+1}/{len(selected)}] {cand.source} {cand.rel_video} "
            f"frames={n} success={cand.success} cat={cand.category}"
        )

    with (out_dir / "meta" / "tasks.jsonl").open("w") as f:
        for task, idx in sorted(tasks.items(), key=lambda x: x[1]):
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    with (out_dir / "meta" / "episodes.jsonl").open("w") as f:
        for row in episodes_meta:
            f.write(json.dumps(row) + "\n")

    with (out_dir / "meta" / "episodes_stats.jsonl").open("w") as f:
        for row in episodes_stats:
            f.write(json.dumps(row) + "\n")

    # total_chunks / total_videos required by lerobot convert_dataset_v21_to_v30
    n_chunks = max(1, (len(selected) + CHUNKS_SIZE - 1) // CHUNKS_SIZE)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda_so100" if n_real else "panda",
        "total_episodes": len(selected),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_chunks": n_chunks,
        "total_videos": len(selected),  # one camera key
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

    write_dataset_card(out_dir, "local/robofac_preview", info, n_sim=n_sim, n_real=n_real)

    manifest = {
        "n_episodes": len(selected),
        "n_sim": n_sim,
        "n_realworld": n_real,
        "total_frames": total_frames,
        "fps": fps,
        "episodes": [
            {
                "episode_index": i,
                "rel_video": c.rel_video,
                "source": c.source,
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


def write_dataset_card(
    out_dir: Path,
    repo_id: str,
    info: dict[str, Any],
    n_sim: int | None = None,
    n_real: int | None = None,
) -> None:
    # Recover counts from existing card/manifest if not passed (e.g. after v3 convert)
    if n_sim is None or n_real is None:
        man = out_dir / "conversion_manifest.json"
        if man.exists():
            m = json.loads(man.read_text())
            n_sim = m.get("n_sim", n_sim)
            n_real = m.get("n_realworld", n_real)
    n_sim = n_sim if n_sim is not None else info["total_episodes"]
    n_real = n_real if n_real is not None else 0

    readme = f"""---
license: apache-2.0
task_categories:
  - robotics
tags:
  - LeRobot
  - LeRobotDataset
  - RoboFAC
  - panda
  - so100
  - simulation
  - realworld
---

# RoboFAC → LeRobot preview (`{repo_id}`)

Small preview subset of [MINT-SJTU/RoboFAC-dataset](https://huggingface.co/datasets/MINT-SJTU/RoboFAC-dataset)
converted to **LeRobot {info.get("codebase_version", "v2.1+")}** for the
[dataset visualizer](https://huggingface.co/spaces/lerobot/visualize_dataset).

| | |
|--|--|
| Episodes | **{info["total_episodes"]}** ({n_sim} sim + {n_real} realworld) |
| Frames | **{info["total_frames"]}** |
| FPS | **{info["fps"]}** |
| Robot | `{info.get("robot_type")}` |
| Camera | `{CAMERA_KEY}` |
| Format | `{info.get("codebase_version")}` |

## Notes

- **Sim**: actions / `observation.state` are 8-D joint targets from RoboFAC
  `ToolsTask_traj.h5` (`pd_joint_pos`). True proprioception was not stored
  (`obs_mode=none`). Task strings include `[success]` / `[failure]`.
- **Realworld** (`so100_*`): video-only source; camera remapped from
  `observation.images.above` → `{CAMERA_KEY}`. Action/state are **zero
  placeholders** (8-D) so the schema stays compatible. Tasks tagged
  `realworld: so100_… [failure]`; `next.success=False` for `*_error` folders.
  Native resolution may differ from sim (preview / visualization only).
- Visualization / format preview only — not a full RoboFAC port.

## Visualize

`https://lerobot-visualize-dataset.hf.space/{repo_id}`

## Reproduce

```bash
python scripts/convert_robofac_to_lerobot.py \\
  --raw-dir data/raw/robofac \\
  --out-dir data/processed/robofac_lerobot_preview \\
  --max-episodes 32 --include-realworld --max-realworld-episodes 18 --to-v30
```
"""
    (out_dir / "README.md").write_text(readme)


def convert_to_v30(out_dir: Path, repo_id: str) -> None:
    """Run official LeRobot v2.1 → v3.0 converter in-place."""
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

    old = Path(str(out_dir) + "_old")
    if old.exists():
        shutil.rmtree(old)

    if manifest_backup is not None:
        (out_dir / "conversion_manifest.json").write_text(manifest_backup)

    info = json.loads((out_dir / "meta" / "info.json").read_text())
    n_sim = n_real = None
    if manifest_backup is not None:
        m = json.loads(manifest_backup)
        n_sim, n_real = m.get("n_sim"), m.get("n_realworld")
    write_dataset_card(out_dir, repo_id, info, n_sim=n_sim, n_real=n_real)


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
    print(
        f"Structural validation OK ({version}): "
        f"{info['total_episodes']} episodes, {info['total_frames']} frames"
    )


def try_lerobot_load(out_dir: Path, repo_id: str) -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("LeRobot not installed; skip LeRobotDataset load check")
        return
    info = json.loads((out_dir / "meta" / "info.json").read_text())
    if info["codebase_version"] != "v3.0":
        print(
            f"codebase_version={info['codebase_version']}; "
            "LeRobot>=0.4 expects v3.0 — skip load (convert with --to-v30)"
        )
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
    p.add_argument("--max-episodes", type=int, default=32, help="Max simulation episodes")
    p.add_argument(
        "--include-realworld",
        action="store_true",
        help="Also sample realworld_data/so100_* video-only episodes",
    )
    p.add_argument(
        "--max-realworld-episodes",
        type=int,
        default=18,
        help="Max realworld episodes when --include-realworld (default: 18 ≈ 3×6 tasks)",
    )
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

    print(f"Scanning sim candidates under {raw_dir} ...")
    qa_tasks = load_qa_task_map(qa_path)
    print(f"QA task descriptions: {len(qa_tasks)}")
    cands = iter_candidates(raw_dir, qa_tasks)
    print(f"Sim candidates with video+traj: {len(cands)}")
    by_cat = defaultdict(int)
    for c in cands:
        by_cat[c.category] += 1
    print("Sim category counts:", dict(by_cat))

    selected = select_diverse(cands, args.max_episodes, args.seed)
    print(f"Selected {len(selected)} sim episodes")

    if args.include_realworld:
        rw_tasks = load_realworld_task_map(raw_dir)
        print(f"Realworld QA task descriptions: {len(rw_tasks)}")
        rw_cands = iter_realworld_candidates(raw_dir, rw_tasks)
        print(f"Realworld candidates: {len(rw_cands)}")
        by_task = defaultdict(int)
        for c in rw_cands:
            by_task[c.task_folder] += 1
        print("Realworld per-task:", dict(by_task))
        rw_selected = select_realworld_diverse(
            rw_cands, args.max_realworld_episodes, args.seed
        )
        print(f"Selected {len(rw_selected)} realworld episodes")
        selected = selected + rw_selected

    print(f"Building combined preview: {len(selected)} episodes")
    info = build_v21(selected, out_dir, fps_override=args.fps)
    if not args.skip_validate:
        structural_validate(out_dir)

    if args.to_v30:
        convert_to_v30(out_dir, args.repo_id)
        if not args.skip_validate:
            structural_validate(out_dir)
            try_lerobot_load(out_dir, args.repo_id)

    print(f"Done. Dataset at {out_dir}")
    print(
        "Visualizer: "
        f"https://lerobot-visualize-dataset.hf.space/{args.repo_id}"
    )


if __name__ == "__main__":
    main()
