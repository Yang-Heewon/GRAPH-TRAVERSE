# Script Guide (Cross-Platform)

This guide lists each shell script and how to run it on Linux/macOS/Windows (Git Bash or WSL).

## Common

- All scripts now auto-detect Python in this order: `PYTHON_BIN` -> `python` -> `python3` -> `py -3`.
- To force a Python launcher:
  - `PYTHON_BIN=python3 bash scripts/setup_and_preprocess.sh`
  - `PYTHON_BIN="py -3" bash scripts/setup_and_preprocess.sh`

## scripts/setup_and_preprocess.sh

- Purpose: Download/map data and run preprocess.
- Run:
  - `bash scripts/setup_and_preprocess.sh`
- Key env:
  - `DATASET=all|cwq|webqsp`
  - `MAX_STEPS`, `MAX_PATHS`, `MINE_MAX_NEIGHBORS`, `PREPROCESS_WORKERS`

## scripts/download_data.sh

- Purpose: Download data from Google Drive or URLs, then map files into expected paths.
- Run:
  - `bash scripts/download_data.sh`
- Key env:
  - `DATASET=all|cwq|webqsp`
  - `SKIP_GDRIVE=1` to disable GDrive
  - `GDRIVE_FILE_URL`, `GDRIVE_FOLDER_URL`, `WEBQSP_URL`, `CWQ_URL`
  - Per-file URLs: `WEBQSP_TRAIN_URL`, `CWQ_TRAIN_URL`, etc.

## scripts/preprocess_cwq_then_webqsp.sh

- Purpose: Run preprocess in order `cwq -> webqsp`.
- Run:
  - `bash scripts/preprocess_cwq_then_webqsp.sh`
- Key env:
  - `MAX_STEPS`, `MAX_PATHS`, `MINE_MAX_NEIGHBORS`, `PREPROCESS_WORKERS`

## scripts/custom_cwq_env.sh

- Purpose: Shared env defaults for custom CWQ pipeline.
- Notes:
  - Defaults are now repo-relative (no machine-specific absolute paths).
  - Override any path with env vars before calling downstream scripts.

## scripts/run_preprocess_custom_cwq.sh

- Purpose: Preprocess custom CWQ data.
- Run:
  - `bash scripts/run_preprocess_custom_cwq.sh`

## scripts/run_embed_custom_cwq.sh

- Purpose: Build embeddings for custom CWQ data.
- Run:
  - `bash scripts/run_embed_custom_cwq.sh`

## scripts/run_train_custom_cwq.sh

- Purpose: Train on custom CWQ data.
- Run:
  - `bash scripts/run_train_custom_cwq.sh`

## scripts/train_with_custom_cwq.sh

- Purpose: Run preprocess + embed + train in sequence.
- Run:
  - `bash scripts/train_with_custom_cwq.sh`

## scripts/run_local_demo.sh

- Purpose: Create tiny demo data and run local end-to-end test.
- Run:
  - `bash scripts/run_local_demo.sh`

## scripts/use_agent_rag_data.sh

- Purpose: Link/copy agent-rag dataset files into this repo layout.
- Run:
  - `bash scripts/use_agent_rag_data.sh`
- Key env:
  - `SRC_ROOT` (default: `$HOME/agent-rag/data`)
  - `DATASET=webqsp|cwq|all`
  - `MODE=symlink|copy`
  - `FORCE=1` to overwrite targets

## trm_agent/scripts/run_preprocess.sh

- Purpose: Preprocess via TRM pipeline.
- Run:
  - `bash trm_agent/scripts/run_preprocess.sh`

## trm_agent/scripts/run_embed.sh

- Purpose: Embedding stage via TRM pipeline.
- Run:
  - `bash trm_agent/scripts/run_embed.sh`

## trm_agent/scripts/run_train.sh

- Purpose: Distributed training entrypoint (`torchrun`).
- Run:
  - `bash trm_agent/scripts/run_train.sh`
- Key env:
  - `TORCHRUN`, `CUDA_VISIBLE_DEVICES`, `DATASET`, `MODEL_IMPL`, `EMB_MODEL`

## trm_agent/scripts/run_test.sh

- Purpose: Evaluate checkpoint.
- Run:
  - `CKPT=/path/to/model_epX.pt bash trm_agent/scripts/run_test.sh`
- Key env:
  - `DATASET`, `MODEL_IMPL`, `CKPT`

## trm_agent/scripts/run_all.sh

- Purpose: Run all pipeline stages (`stage=all`).
- Run:
  - `bash trm_agent/scripts/run_all.sh`
- Key env:
  - `DATASET`, `MODEL_IMPL`, `EMB_MODEL`
