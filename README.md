# GRAPH-TRAVERSE (Subgraph Reader Track)

This repository is organized around a **Subgraph Reader** KGQA pipeline with two reproducible training tracks:

- `v2` baseline (stable BCE-oriented subgraph reader)
- `outer_yz` track (TRM-style outer reasoning loop over `(y, z)`)

The current setup supports:

1. data download / preprocessing
2. embedding generation
3. distributed training (DDP)
4. evaluation on dev/test

## 1) Repo Layout

- `trm_agent/`: dataset jsonl and embeddings/ckpt locations used by runtime
- `trm_unified/`: model and core training logic (`subgraph_reader.py`, `train_core.py`)
- `trm_rag_style/`: run entrypoint, config, and training scripts
- `trm_rag_style/scripts/README.md`: script-level quick guide

## 2) Environment

Recommended Python env: `taiLab`

```bash
cd /data2/workspace/heewon/GRAPH-TRAVERSE-newrepo
pip install -r requirements.txt
```

## 3) Data and Embeddings

### 3.1 Download / preprocess

```bash
bash trm_rag_style/scripts/run_download.sh
```

### 3.2 Build embeddings (entity/relation/query)

```bash
DATASET=cwq \
EMB_MODEL=intfloat/multilingual-e5-large \
EMB_TAG=e5_w4_g4 \
EMBED_GPUS=0,1,2,3 \
EMBED_BATCH_SIZE=512 \
bash trm_rag_style/scripts/run_embed.sh
```

Embeddings are resolved under:

- `trm_agent/emb/${DATASET}_${EMB_TAG}`

## 4) Training Tracks

## 4.1 v2 baseline (recommended strong baseline)

Resume/scratch capable script:

- `trm_rag_style/scripts/run_train_subgraph_v2_resume.sh`

Example:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MASTER_PORT=29606 \
EPOCHS=50 \
BATCH_SIZE=1 \
HIDDEN_SIZE=768 \
SUBGRAPH_RECURSION_STEPS=8 \
SUBGRAPH_MAX_NODES=2048 \
SUBGRAPH_MAX_EDGES=8192 \
EMB_TAG=e5_w4_g4 \
EMB_DIR=trm_agent/emb/cwq_e5_w4_g4 \
CKPT_DIR=trm_agent/ckpt/cwq_trm_hier6_subgraph_3gpu_v2 \
bash trm_rag_style/scripts/run_train_subgraph_v2_resume.sh
```

## 4.2 outer_yz (TRM-style outer reasoning)

Primary script:

- `trm_rag_style/scripts/run_train_subgraph_outer_yz_resume.sh`

What it supports:

- `FROM_SCRATCH=true`: start from random init (`CKPT` ignored)
- `FROM_SCRATCH=false`: resume/fine-tune from checkpoint
- cosine LR schedule defaults for scratch
- gradient accumulation via `SUBGRAPH_GRAD_ACCUM_STEPS`

Example (scratch, cosine annealing, 4-GPU):

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MASTER_PORT=29618 \
FROM_SCRATCH=true \
EPOCHS=50 \
LR=5e-5 \
SUBGRAPH_LR_SCHEDULER=cosine \
SUBGRAPH_LR_MIN=1e-6 \
HIDDEN_SIZE=768 \
SUBGRAPH_RECURSION_STEPS=10 \
SUBGRAPH_OUTER_REASONING_STEPS=2 \
SUBGRAPH_GRAD_ACCUM_STEPS=4 \
WANDB_MODE=online \
WANDB_RUN_NAME=cwq_outer_yz_scratch_ca_lr5e5 \
bash trm_rag_style/scripts/run_train_subgraph_outer_yz_resume.sh
```

## 5) Evaluation

Single checkpoint test:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MASTER_PORT=29620 \
python -m trm_agent.run \
  --dataset cwq \
  --model_impl trm_hier6 \
  --stage test \
  --ckpt trm_agent/ckpt/<your_dir>/model_epXX.pt \
  --override \
    emb_tag=e5_w4_g4 \
    emb_dir=trm_agent/emb/cwq_e5_w4_g4 \
    eval_json=trm_agent/processed/cwq/test.jsonl \
    query_emb_eval_npy=trm_agent/emb/cwq_e5_w4_g4/query_test.npy \
    batch_size=8 \
    subgraph_reader_enabled=true
```

## 6) Notes on Current Training Logic

- Node-level binary objective: BCE-with-logits
- Optional hard-negative BCE filtering (`subgraph_bce_hard_negative_enabled`)
- Optional ranking hard-negative loss (`subgraph_ranking_enabled`)
- Outer reasoning loop (`subgraph_outer_reasoning_enabled`) adds global latent updates
- Gradient accumulation in subgraph training (`subgraph_grad_accum_steps`)

## 7) W&B

If login is already configured (`~/.netrc`), use:

- `WANDB_MODE=online`
- `WANDB_PROJECT=graph-traverse`
- `WANDB_RUN_NAME=<your_run_name>`

