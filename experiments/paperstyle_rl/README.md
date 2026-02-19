# Paperstyle -> RL Pipeline

This folder contains the active preset:

1. `preprocess`
2. `embed` (GNN-RAG exact style)
3. `phase1` supervised train
4. `phase2` RL fine-tune
5. `test` evaluation

## Cross-Platform Entrypoint

Use `run_pipeline.py` on Linux/macOS/Windows.

Linux/macOS:

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE
python experiments/paperstyle_rl/run_pipeline.py --stage all
```

Windows PowerShell:

```powershell
cd C:\path\to\GRAPH-TRAVERSE
.\experiments\paperstyle_rl\run_pipeline.ps1 -Stage all
```

Windows CMD:

```bat
cd C:\path\to\GRAPH-TRAVERSE
experiments\paperstyle_rl\run_pipeline.cmd --stage all
```

Run one stage only:

```bash
python experiments/paperstyle_rl/run_pipeline.py --stage phase1
```

## Linux Bash Wrappers

The existing bash wrappers are still available (`00_*.sh`, `run_all.sh`, `run_all_wandb.sh`) and map to the same stages.

## Key Environment Variables

Main variables:

- `DATASET` (default: `cwq`)
- `EPOCHS_PHASE1` (default: `5`)
- `EPOCHS_PHASE2` (default: `20`)
- `BATCH_SIZE_PHASE1` (default: `6`)
- `BATCH_SIZE_PHASE2` (default: `2`)
- `LR` (default: `2e-4`)
- `NPROC_PER_NODE` (default: `3`)
- `NPROC_PER_NODE_PHASE2` (default: `1`)
- `EMBED_STYLE` (default: `gnn_rag_gnn_exact`)
- `EMBED_BACKEND` (default: `sentence_transformers`)
- `EMB_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
- `WANDB_MODE`, `WANDB_PROJECT`, `WANDB_ENTITY`
- `CKPT_DIR_PHASE1`, `CKPT_DIR_PHASE2`, `RESULTS_DIR`

Test-stage variables:

- `TEST_EVAL_LIMIT`, `TEST_DEBUG_EVAL_N`, `TEST_EVAL_PRED_TOPK`
- `TEST_EVAL_NO_CYCLE`, `TEST_EVAL_USE_HALT`
- `TEST_EVAL_MAX_NEIGHBORS`, `TEST_EVAL_PRUNE_KEEP`
- `TEST_EVAL_BEAM`, `TEST_EVAL_START_TOPK`

## Notes

- In `gnn_rag_gnn_exact` mode, query/passage prefixes are empty.
- Phase2 automatically uses the latest phase1 checkpoint unless `PHASE1_CKPT` is set.
- Test stage uses the latest phase2 checkpoint unless `PHASE2_CKPT` is set.
- Test summaries/logs are under `experiments/paperstyle_rl/results/`.
