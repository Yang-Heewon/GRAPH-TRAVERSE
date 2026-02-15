#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib/portable_env.sh"
PYTHON_BIN="$(require_python_bin)"
cd "$ROOT_DIR"

$PYTHON_BIN scripts/make_demo_webqsp.py

$PYTHON_BIN -m trm_rag_style.run \
  --dataset webqsp \
  --stage all \
  --model_impl trm_hier6 \
  --embedding_model local-hash \
  --override \
    trm_tokenizer=local-hash \
    embed_device=cpu \
    embed_batch_size=16 \
    num_workers=0 \
    batch_size=1 \
    epochs=1 \
    hidden_size=64 \
    num_heads=4 \
    H_cycles=1 \
    L_cycles=2 \
    L_layers=1 \
    eval_limit=20 \
    debug_eval_n=1

CKPT_PATH="$(ls -1 trm_rag_style/ckpt/webqsp_trm_hier6/model_ep*.pt | tail -n 1)"
$PYTHON_BIN -m trm_rag_style.run \
  --dataset webqsp \
  --stage test \
  --model_impl trm_hier6 \
  --embedding_model local-hash \
  --ckpt "$CKPT_PATH" \
  --override \
    trm_tokenizer=local-hash \
    batch_size=1 \
    hidden_size=64 \
    num_heads=4 \
    H_cycles=1 \
    L_cycles=2 \
    L_layers=1 \
    eval_limit=20 \
    debug_eval_n=1

echo "✅ local demo finished"
