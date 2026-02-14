#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

DATASET=${DATASET:-webqsp}
EMB_MODEL=${EMB_MODEL:-intfloat/multilingual-e5-large}
python -m trm_rag_style.run --dataset "$DATASET" --stage embed --embedding_model "$EMB_MODEL"
