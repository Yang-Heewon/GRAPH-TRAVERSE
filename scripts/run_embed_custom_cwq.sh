#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/custom_cwq_env.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/lib/portable_env.sh"
PYTHON_BIN="$(require_python_bin)"

$PYTHON_BIN -m trm_agent.run \
  --dataset cwq \
  --stage embed \
  --model_impl "$MODEL_IMPL" \
  --embedding_model "$EMB_MODEL" \
  --override \
    processed_dir="$PROC_DIR" \
    emb_dir="$EMB_DIR" \
    ckpt_dir="$CKPT_DIR" \
    entities_txt="$ENTITY_TEXT_OUT" \
    relations_txt="$RELATIONS_TXT" \
    entity_names_json="$ENTITY_NAMES_JSON" \
    merged_entities_txt="$ENTITY_TEXT_OUT" \
    custom_train_jsonl="$TRAIN_JSONL" \
    custom_dev_jsonl="$DEV_JSONL" \
    custom_link_mode=symlink

echo "✅ embed done: $EMB_DIR"
