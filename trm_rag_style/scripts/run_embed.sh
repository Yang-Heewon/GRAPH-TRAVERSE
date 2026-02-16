#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/lib/portable_env.sh"
PYTHON_BIN="$(require_python_bin)"
cd "$REPO_ROOT"

DATASET=${DATASET:-webqsp}
EMB_MODEL=${EMB_MODEL:-intfloat/multilingual-e5-large}
EMBED_DEVICE=${EMBED_DEVICE:-cuda}
EMBED_GPUS=${EMBED_GPUS:-}
$PYTHON_BIN -m trm_agent.run --dataset "$DATASET" --stage embed --embedding_model "$EMB_MODEL" \
  --override embed_device="$EMBED_DEVICE" embed_gpus="$EMBED_GPUS"
