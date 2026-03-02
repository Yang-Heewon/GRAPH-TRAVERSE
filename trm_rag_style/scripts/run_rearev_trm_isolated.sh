#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TORCHRUN_BIN="${TORCHRUN_BIN:-/data2/workspace/heewon/anaconda3/envs/taiLab/bin/torchrun}"
DATASET="${DATASET:-cwq}"
MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
EMB_TAG="${EMB_TAG:-e5_w4_g4}"
EMB_DIR="${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

# Keep this run isolated from prior experiments by default.
CKPT_DIR="${CKPT_DIR:-trm_agent/ckpt/${DATASET}_${MODEL_IMPL}_rearev_trm_isolated_${RUN_TAG}}"

EPOCHS="${EPOCHS:-16}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-2e-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29678}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${DATASET}-rearev-trm-isolated-${RUN_TAG}}"

SUBGRAPH_MAX_NODES="${SUBGRAPH_MAX_NODES:-2048}"
SUBGRAPH_MAX_EDGES="${SUBGRAPH_MAX_EDGES:-8192}"
SUBGRAPH_LOSS_MODE="${SUBGRAPH_LOSS_MODE:-rearev_trm}"
SUBGRAPH_RECURSION_STEPS="${SUBGRAPH_RECURSION_STEPS:-4}"
SUBGRAPH_REAREV_ADAPT_STAGES="${SUBGRAPH_REAREV_ADAPT_STAGES:-2}"
SUBGRAPH_GRAD_ACCUM_STEPS="${SUBGRAPH_GRAD_ACCUM_STEPS:-4}"
SUBGRAPH_REAREV_NORMALIZED_GNN="${SUBGRAPH_REAREV_NORMALIZED_GNN:-false}"
SUBGRAPH_LR_SCHEDULER="${SUBGRAPH_LR_SCHEDULER:-cosine}"
SUBGRAPH_LR_MIN="${SUBGRAPH_LR_MIN:-1e-6}"
SUBGRAPH_REAREV_DYNAMIC_HALTING_THRESHOLD="${SUBGRAPH_REAREV_DYNAMIC_HALTING_THRESHOLD:-0.95}"
SUBGRAPH_REAREV_DYNAMIC_HALTING_MIN_STEPS="${SUBGRAPH_REAREV_DYNAMIC_HALTING_MIN_STEPS:-3}"
SUBGRAPH_REAREV_TRM_HALT_BCE_WEIGHT="${SUBGRAPH_REAREV_TRM_HALT_BCE_WEIGHT:-1.0}"
SUBGRAPH_REAREV_TRM_CE_WEIGHT="${SUBGRAPH_REAREV_TRM_CE_WEIGHT:-1.0}"
SUBGRAPH_REAREV_TRM_WEIGHT="${SUBGRAPH_REAREV_TRM_WEIGHT:-1.0}"
SUBGRAPH_REAREV_TRM_STYLE_ENABLED="${SUBGRAPH_REAREV_TRM_STYLE_ENABLED:-true}"
SUBGRAPH_REAREV_TRM_TMINUS1_NO_GRAD="${SUBGRAPH_REAREV_TRM_TMINUS1_NO_GRAD:-true}"
SUBGRAPH_REAREV_TRM_DETACH_CARRY="${SUBGRAPH_REAREV_TRM_DETACH_CARRY:-true}"
# TRM t-1 no-grad uses only the last recursion step for autograd, so many params
# are intentionally unused each iteration. Keep DDP in find-unused mode by default.
SUBGRAPH_DDP_FIND_UNUSED_PARAMETERS="${SUBGRAPH_DDP_FIND_UNUSED_PARAMETERS:-true}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_ANONYMOUS="${WANDB_ANONYMOUS:-allow}"

"$TORCHRUN_BIN" --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
  -m trm_agent.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --stage train \
  --override \
    emb_tag="$EMB_TAG" \
    emb_dir="$EMB_DIR" \
    ckpt_dir="$CKPT_DIR" \
    epochs="$EPOCHS" \
    batch_size="$BATCH_SIZE" \
    lr="$LR" \
    eval_every_epochs=1 \
    eval_start_epoch=1 \
    eval_limit=-1 \
    wandb_mode="$WANDB_MODE" \
    wandb_project="$WANDB_PROJECT" \
    wandb_run_name="$WANDB_RUN_NAME" \
    subgraph_reader_enabled=true \
    subgraph_loss_mode="$SUBGRAPH_LOSS_MODE" \
    subgraph_max_nodes="$SUBGRAPH_MAX_NODES" \
    subgraph_max_edges="$SUBGRAPH_MAX_EDGES" \
    subgraph_add_reverse_edges=true \
    subgraph_split_reverse_relations=true \
    subgraph_direction_embedding_enabled=true \
    subgraph_rearev_num_ins=3 \
    subgraph_rearev_adapt_stages="$SUBGRAPH_REAREV_ADAPT_STAGES" \
    subgraph_recursion_steps="$SUBGRAPH_RECURSION_STEPS" \
    subgraph_grad_accum_steps="$SUBGRAPH_GRAD_ACCUM_STEPS" \
    subgraph_rearev_normalized_gnn="$SUBGRAPH_REAREV_NORMALIZED_GNN" \
    subgraph_lr_scheduler="$SUBGRAPH_LR_SCHEDULER" \
    subgraph_lr_min="$SUBGRAPH_LR_MIN" \
    subgraph_rearev_latent_reasoning_enabled=true \
    subgraph_rearev_latent_residual_alpha=0.25 \
    subgraph_rearev_global_gate_enabled=true \
    subgraph_rearev_logit_global_fusion_enabled=true \
    subgraph_rearev_dynamic_halting_enabled=true \
    subgraph_rearev_dynamic_halting_min_steps="$SUBGRAPH_REAREV_DYNAMIC_HALTING_MIN_STEPS" \
    subgraph_rearev_dynamic_halting_threshold="$SUBGRAPH_REAREV_DYNAMIC_HALTING_THRESHOLD" \
    subgraph_rearev_trm_style_enabled="$SUBGRAPH_REAREV_TRM_STYLE_ENABLED" \
    subgraph_rearev_trm_tminus1_no_grad="$SUBGRAPH_REAREV_TRM_TMINUS1_NO_GRAD" \
    subgraph_rearev_trm_detach_carry="$SUBGRAPH_REAREV_TRM_DETACH_CARRY" \
    subgraph_rearev_trm_halt_bce_weight="$SUBGRAPH_REAREV_TRM_HALT_BCE_WEIGHT" \
    subgraph_rearev_trm_ce_weight="$SUBGRAPH_REAREV_TRM_CE_WEIGHT" \
    subgraph_rearev_trm_weight="$SUBGRAPH_REAREV_TRM_WEIGHT" \
    subgraph_ddp_find_unused_parameters="$SUBGRAPH_DDP_FIND_UNUSED_PARAMETERS"
