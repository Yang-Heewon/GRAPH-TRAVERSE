#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATASET=${DATASET:-webqsp}
MODEL_NAME=${MODEL_NAME:-intfloat/multilingual-e5-large}
if [ "$DATASET" = "cwq" ]; then
  python -m trm_unified.pipeline embed \
    --model_name "$MODEL_NAME" \
    --entities_txt data/CWQ/embeddings_output/CWQ/e5/entity_ids.txt \
    --relations_txt data/CWQ/embeddings_output/CWQ/e5/relation_ids.txt \
    --train_jsonl trm_unified/processed/cwq/train.jsonl \
    --dev_jsonl trm_unified/processed/cwq/dev.jsonl \
    --out_dir trm_unified/emb/cwq
else
  python -m trm_unified.pipeline embed \
    --model_name "$MODEL_NAME" \
    --entities_txt data/webqsp/entities.txt \
    --relations_txt data/webqsp/relations.txt \
    --train_jsonl trm_unified/processed/webqsp/train.jsonl \
    --dev_jsonl trm_unified/processed/webqsp/dev.jsonl \
    --out_dir trm_unified/emb/webqsp
fi
