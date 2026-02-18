#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Core run settings
export DATASET="${DATASET:-cwq}"
export MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
export EPOCHS="${EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-6}"
export LR="${LR:-1e-4}"
export MAX_STEPS="${MAX_STEPS:-6}"
export EVAL_MAX_STEPS="${EVAL_MAX_STEPS:-6}"
export EVAL_EVERY_EPOCHS="${EVAL_EVERY_EPOCHS:-2}"
export EVAL_START_EPOCH="${EVAL_START_EPOCH:-2}"
export EVAL_LIMIT="${EVAL_LIMIT:-200}"
export EVAL_PRED_TOPK="${EVAL_PRED_TOPK:-1}"
export EVAL_USE_HALT="${EVAL_USE_HALT:-false}"
export EVAL_NO_CYCLE="${EVAL_NO_CYCLE:-true}"

# Distributed/GPU settings
export TORCHRUN="${TORCHRUN:-/data2/workspace/heewon/anaconda3/envs/taiLab/bin/torchrun}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-3}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2}"
export MASTER_PORT="${MASTER_PORT:-29631}"
export DDP_FIND_UNUSED="${DDP_FIND_UNUSED:-true}"

# Start from scratch
export CKPT="${CKPT:-}"
export CKPT_DIR="${CKPT_DIR:-trm_rag_style/ckpt/${DATASET}_${MODEL_IMPL}_2phase_wandb}"

# W&B online
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"
export WANDB_ENTITY="${WANDB_ENTITY:-heewon6205-chung-ang-university}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${DATASET}_2phase_wandb_ep${EPOCHS}}"

# Phase-1 objective
export ENDPOINT_LOSS_MODE="${ENDPOINT_LOSS_MODE:-metric_align_main}"
export RELATION_AUX_WEIGHT="${RELATION_AUX_WEIGHT:-0.2}"
export ENDPOINT_AUX_WEIGHT="${ENDPOINT_AUX_WEIGHT:-0.05}"
export METRIC_ALIGN_AUX_WEIGHT="${METRIC_ALIGN_AUX_WEIGHT:-0.0}"
export HALT_AUX_WEIGHT="${HALT_AUX_WEIGHT:-0.05}"

# Phase-2 objective (turn off aux terms to align closer to endpoint metric)
export PHASE2_START_EPOCH="${PHASE2_START_EPOCH:-11}"
export PHASE2_ENDPOINT_LOSS_MODE="${PHASE2_ENDPOINT_LOSS_MODE:-metric_align_main}"
export PHASE2_RELATION_AUX_WEIGHT="${PHASE2_RELATION_AUX_WEIGHT:-0}"
export PHASE2_ENDPOINT_AUX_WEIGHT="${PHASE2_ENDPOINT_AUX_WEIGHT:-0}"
export PHASE2_METRIC_ALIGN_AUX_WEIGHT="${PHASE2_METRIC_ALIGN_AUX_WEIGHT:-0}"
export PHASE2_HALT_AUX_WEIGHT="${PHASE2_HALT_AUX_WEIGHT:-0}"

# Keep auto-switch disabled in this preset (uses fixed 2-phase by epoch boundary)
export PHASE2_AUTO_ENABLED="${PHASE2_AUTO_ENABLED:-false}"

bash trm_rag_style/scripts/run_train.sh

