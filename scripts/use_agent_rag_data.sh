#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${SRC_ROOT:-$HOME/agent-rag/data}"
DATASET="${DATASET:-webqsp}"         # webqsp | cwq | all
MODE="${MODE:-symlink}"              # symlink | copy
FORCE="${FORCE:-0}"                  # 1 to overwrite existing targets

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

safe_link_or_copy() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$dst")"

  if [[ ! -e "$src" ]]; then
    echo "❌ missing source: $src"
    exit 1
  fi

  if [[ -e "$dst" || -L "$dst" ]]; then
    if [[ "$FORCE" != "1" ]]; then
      echo "⚠️ target exists, skip (set FORCE=1 to overwrite): $dst"
      return 0
    fi
    rm -rf "$dst"
  fi

  if [[ "$MODE" == "copy" ]]; then
    cp -a "$src" "$dst"
  else
    ln -s "$src" "$dst"
  fi
  echo "✅ $dst"
}

setup_webqsp() {
  local src="$SRC_ROOT/webqsp"
  local dst="$ROOT_DIR/data/webqsp"
  mkdir -p "$dst"
  safe_link_or_copy "$src/train_simple.json" "$dst/train.json"
  safe_link_or_copy "$src/dev_simple.json" "$dst/dev.json"
  safe_link_or_copy "$src/test_simple.json" "$dst/test.json"
  safe_link_or_copy "$src/entities.txt" "$dst/entities.txt"
  safe_link_or_copy "$src/relations.txt" "$dst/relations.txt"
}

setup_cwq() {
  local src="$SRC_ROOT/CWQ"
  local dst="$ROOT_DIR/data/CWQ"
  mkdir -p "$dst/embeddings_output/CWQ/e5"
  safe_link_or_copy "$src/train_paths_supervised.json" "$dst/train_split.jsonl"
  safe_link_or_copy "$src/dev_converted.json" "$dst/dev_split.jsonl"
  safe_link_or_copy "$src/test_simple.json" "$dst/test_split.jsonl"
  safe_link_or_copy "$src/entities.txt" "$dst/embeddings_output/CWQ/e5/entity_ids.txt"
  safe_link_or_copy "$src/relations.txt" "$dst/embeddings_output/CWQ/e5/relation_ids.txt"
}

dataset_lc="$(printf '%s' "$DATASET" | tr '[:upper:]' '[:lower:]')"

case "$dataset_lc" in
  webqsp)
    setup_webqsp
    ;;
  cwq)
    setup_cwq
    ;;
  all)
    setup_webqsp
    setup_cwq
    ;;
  *)
    echo "❌ DATASET must be one of: webqsp | cwq | all"
    exit 1
    ;;
esac

echo "Done. DATASET=$DATASET MODE=$MODE SRC_ROOT=$SRC_ROOT"
