#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/lib/portable_env.sh"
PYTHON_BIN="$(require_python_bin)"
cd "$REPO_ROOT"

DATASET=${DATASET:-webqsp}
MODEL_IMPL=${MODEL_IMPL:-trm_hier6}
EMB_MODEL=${EMB_MODEL:-intfloat/multilingual-e5-large}

$PYTHON_BIN -m trm_agent.run \
  --dataset "$DATASET" \
  --model_impl "$MODEL_IMPL" \
  --embedding_model "$EMB_MODEL" \
  --stage all
