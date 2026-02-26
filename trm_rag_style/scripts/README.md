# Script Guide (Subgraph Reader)

This folder includes multiple experiment scripts. For current usage, focus on the scripts below.

## Recommended Core Scripts

- `run_download.sh`
  - Download/prepare dataset files.

- `run_embed.sh`
  - Build entity/relation/query embeddings.

- `run_train_subgraph_v2_resume.sh`
  - v2 subgraph baseline training (stable default path).
  - Supports resume via `CKPT` and `SUBGRAPH_RESUME_EPOCH`.

- `run_train_subgraph_outer_yz_resume.sh`
  - TRM-like outer reasoning (`outer_yz`) training profile.
  - Supports:
    - `FROM_SCRATCH=true` (fresh training)
    - `FROM_SCRATCH=false` (resume fine-tune)
    - cosine scheduling + gradient accumulation controls.

- `run_test.sh`
  - Generic test entrypoint.

## Utilities

- `run_select_subgraph_ckpt_devtest_gap.sh`
  - Evaluate ep-range checkpoints and rank by dev/test consistency.

- `run_select_and_finetune_subgraph.sh`
  - Selection + fine-tuning pipeline helper.

## Important Env Vars

- Data/embedding:
  - `DATASET`, `EMB_MODEL`, `EMB_TAG`, `EMB_DIR`
- DDP/runtime:
  - `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE`, `MASTER_PORT`
- Subgraph model:
  - `HIDDEN_SIZE`, `SUBGRAPH_RECURSION_STEPS`, `SUBGRAPH_OUTER_REASONING_STEPS`
  - `SUBGRAPH_MAX_NODES`, `SUBGRAPH_MAX_EDGES`
  - `SUBGRAPH_RANKING_ENABLED`, `SUBGRAPH_BCE_HARD_NEGATIVE_ENABLED`
  - `SUBGRAPH_GRAD_ACCUM_STEPS`
- Optimization:
  - `LR`, `SUBGRAPH_LR_SCHEDULER`, `SUBGRAPH_LR_MIN`

## Typical Start

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE-newrepo
bash trm_rag_style/scripts/run_train_subgraph_outer_yz_resume.sh
```

Then override env vars as needed.
