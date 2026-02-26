#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-cwq}"
MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
EMB_MODEL="${EMB_MODEL:-intfloat/multilingual-e5-large}"
EMB_TAG="${EMB_TAG:-$(echo "$EMB_MODEL" | tr '/:' '__' | tr -cd '[:alnum:]_.-')}"
EMB_DIR="${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}"
CKPT_DIR="${CKPT_DIR:-trm_agent/ckpt/${DATASET}_${MODEL_IMPL}_subgraph_r16_h768_schedcos_scratch}"

TARGET_EP="${TARGET_EP:-28}"
WAIT_INTERVAL_SEC="${WAIT_INTERVAL_SEC:-120}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-1440}"
TRAIN_PROC_PATTERN="${TRAIN_PROC_PATTERN:-cwq_r16_h768_schedcos_scratch}"

EP_START="${EP_START:-20}"
EP_END="${EP_END:-28}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29612}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-cwq fine_tune}"
WANDB_PROJECT="${WANDB_PROJECT:-graph-traverse}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

FT_EPOCHS="${FT_EPOCHS:-10}"
FT_LR="${FT_LR:-1e-6}"

target_ckpt="${CKPT_DIR}/model_ep${TARGET_EP}.pt"
max_wait_sec=$((MAX_WAIT_MIN * 60))
waited=0

echo "[wait] target ckpt: $target_ckpt"
echo "[wait] train proc pattern: $TRAIN_PROC_PATTERN"
echo "[wait] max wait: ${MAX_WAIT_MIN} min"

while true; do
  ckpt_ready=0
  proc_running=0

  if [ -f "$target_ckpt" ]; then
    ckpt_ready=1
  fi
  if ps -eo cmd | grep -F "$TRAIN_PROC_PATTERN" | grep -v grep >/dev/null 2>&1; then
    proc_running=1
  fi

  if [ "$ckpt_ready" -eq 1 ] && [ "$proc_running" -eq 0 ]; then
    echo "[wait] ready: target ckpt exists and train process not running"
    break
  fi

  echo "[wait] ckpt_ready=$ckpt_ready proc_running=$proc_running elapsed=${waited}s"
  sleep "$WAIT_INTERVAL_SEC"
  waited=$((waited + WAIT_INTERVAL_SEC))
  if [ "$waited" -ge "$max_wait_sec" ]; then
    echo "[err] timeout while waiting. ckpt_ready=$ckpt_ready proc_running=$proc_running"
    exit 2
  fi
done

echo "[run] select(test/dev report) + hard-loss fine-tune"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
NPROC_PER_NODE="$NPROC_PER_NODE" \
MASTER_PORT="$MASTER_PORT" \
WANDB_MODE="$WANDB_MODE" \
WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_ENTITY="$WANDB_ENTITY" \
WANDB_RUN_NAME="$WANDB_RUN_NAME" \
DATASET="$DATASET" \
MODEL_IMPL="$MODEL_IMPL" \
EMB_MODEL="$EMB_MODEL" \
EMB_TAG="$EMB_TAG" \
EMB_DIR="$EMB_DIR" \
CKPT_DIR="$CKPT_DIR" \
EP_START="$EP_START" \
EP_END="$EP_END" \
FT_EPOCHS="$FT_EPOCHS" \
FT_LR="$FT_LR" \
bash trm_rag_style/scripts/run_select_and_finetune_subgraph.sh

echo "[done] wait+select+finetune completed"
