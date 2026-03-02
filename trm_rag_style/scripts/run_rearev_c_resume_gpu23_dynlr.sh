#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

infer_resume_epoch() {
  local ckpt_path="$1"
  local b
  b="$(basename "$ckpt_path")"
  if [[ "$b" =~ model_ep([0-9]+)\.pt$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo "0"
  fi
}

TORCHRUN_BIN="${TORCHRUN_BIN:-/data2/workspace/heewon/anaconda3/envs/taiLab/bin/torchrun}"
DATASET="${DATASET:-cwq}"
MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
EMB_TAG="${EMB_TAG:-e5_w4_g4}"
EMB_DIR="${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}"
CKPT_DIR="${CKPT_DIR:-trm_agent/ckpt/cwq_trm_hier6_rearev_C_gpu123_b2_gatefusionhalt}"
CKPT="${CKPT:-${CKPT_DIR}/model_ep13.pt}"

# additional epochs after resume checkpoint
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-2e-4}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29672}"

WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-${DATASET}-rearev-C-resume-gpu23-dynlr}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-}" # e.g. must | allow | never

SUBGRAPH_RESUME_EPOCH="${SUBGRAPH_RESUME_EPOCH:-$(infer_resume_epoch "$CKPT")}"
SUBGRAPH_RECURSION_STEPS="${SUBGRAPH_RECURSION_STEPS:-4}"
SUBGRAPH_REAREV_ADAPT_STAGES="${SUBGRAPH_REAREV_ADAPT_STAGES:-2}"
SUBGRAPH_GRAD_ACCUM_STEPS="${SUBGRAPH_GRAD_ACCUM_STEPS:-4}"
SUBGRAPH_LR_SCHEDULER="${SUBGRAPH_LR_SCHEDULER:-cosine}"
SUBGRAPH_LR_MIN="${SUBGRAPH_LR_MIN:-1e-6}"
SUBGRAPH_LR_STEP_SIZE="${SUBGRAPH_LR_STEP_SIZE:-5}"
SUBGRAPH_LR_GAMMA="${SUBGRAPH_LR_GAMMA:-0.5}"
SUBGRAPH_LR_PLATEAU_FACTOR="${SUBGRAPH_LR_PLATEAU_FACTOR:-0.5}"
SUBGRAPH_LR_PLATEAU_PATIENCE="${SUBGRAPH_LR_PLATEAU_PATIENCE:-2}"
SUBGRAPH_LR_PLATEAU_THRESHOLD="${SUBGRAPH_LR_PLATEAU_THRESHOLD:-1e-4}"
SUBGRAPH_LR_PLATEAU_METRIC="${SUBGRAPH_LR_PLATEAU_METRIC:-dev_hit1}"
SUBGRAPH_REAREV_DYNAMIC_HALTING_THRESHOLD="${SUBGRAPH_REAREV_DYNAMIC_HALTING_THRESHOLD:-0.95}"
SUBGRAPH_REAREV_DYNAMIC_HALTING_MIN_STEPS="${SUBGRAPH_REAREV_DYNAMIC_HALTING_MIN_STEPS:-3}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export WANDB_ANONYMOUS="${WANDB_ANONYMOUS:-allow}"
if [[ -n "$WANDB_RUN_ID" ]]; then
  export WANDB_RUN_ID
fi
if [[ -n "$WANDB_RESUME" ]]; then
  export WANDB_RESUME
fi

if [[ ! -f "$CKPT" ]]; then
  echo "[err] resume checkpoint not found: $CKPT"
  exit 2
fi

"$TORCHRUN_BIN" --nproc_per_node="$NPROC_PER_NODE" --master_port="$MASTER_PORT" \
  -m trm_agent.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --stage train \
  --ckpt "$CKPT" \
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
    wandb_entity="$WANDB_ENTITY" \
    wandb_run_name="$WANDB_RUN_NAME" \
    subgraph_reader_enabled=true \
    subgraph_loss_mode=rearev_kl \
    subgraph_add_reverse_edges=true \
    subgraph_split_reverse_relations=true \
    subgraph_direction_embedding_enabled=true \
    subgraph_rearev_num_ins=3 \
    subgraph_recursion_steps="$SUBGRAPH_RECURSION_STEPS" \
    subgraph_rearev_adapt_stages="$SUBGRAPH_REAREV_ADAPT_STAGES" \
    subgraph_grad_accum_steps="$SUBGRAPH_GRAD_ACCUM_STEPS" \
    subgraph_lr_scheduler="$SUBGRAPH_LR_SCHEDULER" \
    subgraph_lr_min="$SUBGRAPH_LR_MIN" \
    subgraph_lr_step_size="$SUBGRAPH_LR_STEP_SIZE" \
    subgraph_lr_gamma="$SUBGRAPH_LR_GAMMA" \
    subgraph_lr_plateau_factor="$SUBGRAPH_LR_PLATEAU_FACTOR" \
    subgraph_lr_plateau_patience="$SUBGRAPH_LR_PLATEAU_PATIENCE" \
    subgraph_lr_plateau_threshold="$SUBGRAPH_LR_PLATEAU_THRESHOLD" \
    subgraph_lr_plateau_metric="$SUBGRAPH_LR_PLATEAU_METRIC" \
    subgraph_resume_epoch="$SUBGRAPH_RESUME_EPOCH" \
    subgraph_rearev_latent_reasoning_enabled=true \
    subgraph_rearev_latent_residual_alpha=0.25 \
    subgraph_rearev_global_gate_enabled=true \
    subgraph_rearev_logit_global_fusion_enabled=true \
    subgraph_rearev_dynamic_halting_enabled=true \
    subgraph_rearev_dynamic_halting_threshold="$SUBGRAPH_REAREV_DYNAMIC_HALTING_THRESHOLD" \
    subgraph_rearev_dynamic_halting_min_steps="$SUBGRAPH_REAREV_DYNAMIC_HALTING_MIN_STEPS"
