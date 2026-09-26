# LeRobot visualization path (RoboFAC and similar)

Goal: turn an episode-style robot dataset (MP4 + optional trajectories/QA) into a
**LeRobot Hub repo** that opens in
[lerobot/visualize_dataset](https://huggingface.co/spaces/lerobot/visualize_dataset).

The Space supports **v2.0 / v2.1 / v3.0**. Current `lerobot` Python packages
(≥0.4) prefer **v3.0**; this repo’s converter writes v2.1 then optionally converts
to v3.0 with the official script.

## What the visualizer needs

Minimum viable layout (v2.1, one camera):

```
meta/info.json
meta/episodes.jsonl
meta/tasks.jsonl
meta/episodes_stats.jsonl
data/chunk-000/episode_000000.parquet
videos/chunk-000/observation.images.main/episode_000000.mp4
```

v3.0 consolidates shards:

```
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
data/chunk-000/file-000.parquet
videos/observation.images.main/chunk-000/file-000.mp4
```

Required frame columns (parquet): `timestamp`, `frame_index`, `episode_index`,
`index`, `task_index`, plus `action` / `observation.state` (float vectors) and a
`dtype: video` camera feature in `info.json`. Prefer **H.264 / avc1** MP4s.

Deep link after upload:

`https://lerobot-visualize-dataset.hf.space/<org>/<dataset>`

(Note: `huggingface.co/spaces/lerobot/visualize_dataset/<org>/<dataset>` returns 404; use the `.hf.space` host, or open the Space and paste the repo id.)

## RoboFAC preview (this repo)

Source (gitignored): `data/raw/robofac/`  
Output (gitignored): `data/processed/robofac_lerobot_preview/`

```bash
# Env with lerobot + dataset extras + h5py
source /workspace/.venv-lerobot/bin/activate   # or your own venv
pip install 'lerobot[dataset]' h5py

cd /path/to/robot-deployment-intelligence
python scripts/convert_robofac_to_lerobot.py \
  --raw-dir data/raw/robofac \
  --out-dir data/processed/robofac_lerobot_preview \
  --max-episodes 32 \
  --include-realworld --max-realworld-episodes 18 \
  --seed 42 \
  --repo-id cedricxie/robofac-lerobot-preview \
  --to-v30
```

What the script does:

1. Scans `simulation_data/**/ToolsTask_traj.{json,h5}` + matching `{unique_id}.mp4`.
2. Selects a diverse sim mix (success / forget-* / gripper_error / other failure).
3. With `--include-realworld`, also samples `realworld_data/so100_*/…/observation.images.above`
   evenly across tasks (~3×6 = 18 by default). Remaps that camera → `observation.images.main`,
   fills 8-D **zero** action/state, tags tasks `realworld: so100_… [failure]`, sets
   `next.success=False` for `*_error`, and re-encodes AV1 → H.264 for Hub playback.
4. Writes a **combined** LeRobot **v2.1** preview (same feature schema for sim + real).
5. With `--to-v30`, runs `python -m lerobot.scripts.convert_dataset_v21_to_v30`.
6. Validates structure and tries `LeRobotDataset(...)` load.

Sim task text comes from `training_qa.json` when available; each string is tagged
`[success]` / `[failure]`. Realworld NL hints come from `test_qa_realworld/` when present.
Native realworld resolution (e.g. 640×480) may differ from sim (1024×1024) — preview only.

## Upload to Hugging Face Hub

```bash
# one-time auth (do NOT paste tokens into chat)
huggingface-cli login
# or: export HF_TOKEN=hf_...

# upload local folder as a dataset repo
huggingface-cli upload cedricxie/robofac-lerobot-preview \
  data/processed/robofac_lerobot_preview \
  --repo-type dataset

# or via Python
python - <<'PY'
from pathlib import Path
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('cedricxie/robofac-lerobot-preview', repo_type='dataset', exist_ok=True)
api.upload_folder(
    folder_path='data/processed/robofac_lerobot_preview',
    repo_id='cedricxie/robofac-lerobot-preview',
    repo_type='dataset',
)
PY
```

Then open:

https://lerobot-visualize-dataset.hf.space/cedricxie/robofac-lerobot-preview

If Grok Bot needs a write token later, use the secret-request flow rather than
pasting `HF_TOKEN` into chat.

## Extending to other datasets

| Source field | LeRobot target |
|--------------|----------------|
| Episode MP4 | `videos/.../{camera_key}/...` + `features[camera_key].dtype=video` |
| Joint / EE action | `action` float32 `[D]` |
| Proprioception | `observation.state` float32 `[D]` (zeros OK for video-only preview) |
| Language / VQA | `tasks.jsonl` / episode `tasks` list; optional language features in v3.1 |
| Success label | Encode in task string or `next.success` column |

Tips:

- Keep camera keys like `observation.images.*`.
- One fps for the whole dataset; align parquet length to action length (or video).
- Do **not** commit large processed videos into git — keep `data/processed/` gitignored.
- For large ports, prefer writing with `LeRobotDataset.create()` / `add_frame` /
  `save_episode` directly into v3.0 instead of the v2.1 bridge.

## Local validation

```bash
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(
    repo_id='cedricxie/robofac-lerobot-preview',
    root='data/processed/robofac_lerobot_preview',
    download_videos=False,
)
print(ds.num_episodes, ds.num_frames, ds[0].keys())
PY
```
