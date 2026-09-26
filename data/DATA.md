# Dataset inventory

## Required for MVP

| Dataset | Role | Status |
|---------|------|--------|
| **RoboFAC** (`MINT-SJTU/RoboFAC-dataset`) | Primary development + eval GT (success/failure, VQA/diagnosis text) | **Partial download ready** |
| DROID failure / DROID | External validation (post-MVP) | Not downloaded |
| Guardian / FailCoT | Optional synthetic/external later | Not downloaded |

Full RoboFAC on Hugging Face is ~**25.8 GB**. We intentionally keep a working subset under `data/raw/robofac/` (gitignored).

## What is on disk now

Path: `data/raw/robofac/`

- `README.md` — dataset card
- `training_qa.json` — ~64k VQA items (video path + conversations); used as label/signal source
- `mvp_subset_index.json` — **1200** unique video paths sampled across tasks (seed=42) for the MVP episode list
- `test_qa_sim/annos_per_video_split{0,1,2}.json` — eval-style annotations
- `simulation_data/MicrowaveTask-fork/...` — several leaf folders with `ToolsTask_traj.json`, `.h5`, and sample `.mp4`
- Extra diversity leaves (success / PickCube / StackCube) if download succeeded
- `realworld_data/so100_pick_cube_in_box_error/...` — 10 real-world sample videos

Lightweight copies for adapters/tests: `data/samples/robofac/`

## How to expand later

```bash
# from repo root, with huggingface_hub installed
python -c "from huggingface_hub import snapshot_download; snapshot_download('MINT-SJTU/RoboFAC-dataset', repo_type='dataset', local_dir='data/raw/robofac', allow_patterns=['simulation_data/PickCube-v1/**','training_qa.json'])"
```

Or set `HF_TOKEN` for higher rate limits.

## Notes for adapter (M0)

- Video path in QA: `{task}/{scenario}/{uuid}.mp4`
- Leaf dirs contain `.json` (episode ids) + `.h5` (kinematics by `episode_id`) + `.mp4` (`unique_id`)
- Success trajectories live under tasks like `dataset_success_cleaned`
- Treat missing diagnosis fields as null; detection labels come from success vs error trajectory sets + QA answers


## LeRobot visualization preview

See [`docs/lerobot-visualization.md`](../docs/lerobot-visualization.md). Converter: `scripts/convert_robofac_to_lerobot.py`. Local output (gitignored): `data/processed/robofac_lerobot_preview/` (LeRobot v3.0, ~32 episodes). Hub target: `cedricxie/robofac-lerobot-preview`.
