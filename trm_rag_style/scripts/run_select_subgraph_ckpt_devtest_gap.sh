#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/lib/portable_env.sh"
PYTHON_BIN="$(require_python_bin)"
cd "$REPO_ROOT"

DATASET="${DATASET:-cwq}"
MODEL_IMPL="${MODEL_IMPL:-trm_hier6}"
EMB_MODEL="${EMB_MODEL:-intfloat/multilingual-e5-large}"
EMB_TAG="${EMB_TAG:-$(echo "$EMB_MODEL" | tr '/:' '__' | tr -cd '[:alnum:]_.-')}"
EMB_DIR="${EMB_DIR:-trm_agent/emb/${DATASET}_${EMB_TAG}}"
CKPT_DIR="${CKPT_DIR:-trm_agent/ckpt/${DATASET}_${MODEL_IMPL}_subgraph_r16_h768_schedcos_scratch}"

EP_START="${EP_START:-20}"
EP_END="${EP_END:-27}"
EVAL_LIMIT="${EVAL_LIMIT:--1}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
GAP_TOL="${GAP_TOL:-0.01}"
TEST_NPROC_PER_NODE="${TEST_NPROC_PER_NODE:-1}"
TEST_MASTER_PORT_BASE="${TEST_MASTER_PORT_BASE:-29640}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"

SUBGRAPH_HOPS="${SUBGRAPH_HOPS:-3}"
SUBGRAPH_MAX_NODES="${SUBGRAPH_MAX_NODES:-2048}"
SUBGRAPH_MAX_EDGES="${SUBGRAPH_MAX_EDGES:-8192}"
SUBGRAPH_PRED_THRESHOLD="${SUBGRAPH_PRED_THRESHOLD:-0.5}"
SUBGRAPH_SPLIT_REVERSE_RELATIONS="${SUBGRAPH_SPLIT_REVERSE_RELATIONS:-false}"
SUBGRAPH_DIRECTION_EMBEDDING_ENABLED="${SUBGRAPH_DIRECTION_EMBEDDING_ENABLED:-false}"

OUT_DIR="${OUT_DIR:-logs}"
OUT_CSV="${OUT_CSV:-$OUT_DIR/ckpt_devtest_gap_ep${EP_START}_${EP_END}.csv}"
BEST_ENV="${BEST_ENV:-$OUT_DIR/best_ckpt_ep${EP_START}_${EP_END}.env}"
OUT_REPORT="${OUT_REPORT:-$OUT_DIR/ckpt_devtest_gap_ep${EP_START}_${EP_END}.md}"
STREAM_EVAL_LOGS="${STREAM_EVAL_LOGS:-true}"
DEV_USES_TEST_SPLIT="${DEV_USES_TEST_SPLIT:-false}"
TEST_ONLY_EVAL="${TEST_ONLY_EVAL:-false}"
SELECT_MODE="${SELECT_MODE:-gap_then_test_hit}"  # gap_then_test_hit|test_hit_then_f1|test_f1_then_hit|test_sum

mkdir -p "$OUT_DIR"

if [ ! -f "$EMB_DIR/entity_embeddings.npy" ] || [ ! -f "$EMB_DIR/relation_embeddings.npy" ] || [ ! -f "$EMB_DIR/query_dev.npy" ] || [ ! -f "$EMB_DIR/query_test.npy" ]; then
  echo "[err] required embedding files are missing in $EMB_DIR"
  echo "      expected: entity_embeddings.npy, relation_embeddings.npy, query_dev.npy, query_test.npy"
  exit 2
fi

if [ "$EP_START" -gt "$EP_END" ]; then
  echo "[err] EP_START must be <= EP_END (got $EP_START > $EP_END)"
  exit 2
fi

abs_gap() {
  local a="$1"
  local b="$2"
  awk -v x="$a" -v y="$b" 'BEGIN{d=x-y; if(d<0)d=-d; printf "%.6f", d}'
}

parse_metric_from_log() {
  local log_file="$1"
  local hit
  local f1
  hit="$(grep -oE '\[Test-Subgraph\] Hit@1=[0-9.]+' "$log_file" | tail -n1 | sed -E 's/.*Hit@1=([0-9.]+)/\1/')"
  f1="$(grep -oE 'F1=[0-9.]+' "$log_file" | tail -n1 | sed -E 's/F1=([0-9.]+)/\1/')"
  if [ -z "$hit" ] || [ -z "$f1" ]; then
    echo "[err] failed to parse metrics from $log_file"
    tail -n 40 "$log_file" || true
    return 1
  fi
  echo "$hit,$f1"
}

run_eval_split() {
  local ckpt="$1"
  local split="$2"
  local log_file="$3"
  local ep="$4"
  local eval_json
  local query_npy
  if [ "$split" = "dev" ] && [ "$DEV_USES_TEST_SPLIT" = "true" ]; then
    eval_json="trm_agent/processed/${DATASET}/test.jsonl"
    query_npy="${EMB_DIR}/query_test.npy"
  elif [ "$split" = "dev" ]; then
    eval_json="trm_agent/processed/${DATASET}/dev.jsonl"
    query_npy="${EMB_DIR}/query_dev.npy"
  else
    eval_json="trm_agent/processed/${DATASET}/test.jsonl"
    query_npy="${EMB_DIR}/query_test.npy"
  fi

  local split_offset=0
  if [ "$split" = "test" ]; then
    split_offset=1
  fi
  local run_port=$((TEST_MASTER_PORT_BASE + (ep * 10) + split_offset))
  local -a run_cmd

  if [ "$TEST_NPROC_PER_NODE" -gt 1 ]; then
    run_cmd=(
      "$TORCHRUN_BIN"
      --nproc_per_node="$TEST_NPROC_PER_NODE"
      --master_port="$run_port"
      -m trm_agent.run
      --dataset "$DATASET"
      --model_impl "$MODEL_IMPL"
      --stage test
      --ckpt "$ckpt"
      --override
      emb_tag="$EMB_TAG"
      emb_dir="$EMB_DIR"
      eval_json="$eval_json"
      query_emb_eval_npy="$query_npy"
      eval_limit="$EVAL_LIMIT"
      debug_eval_n=0
      batch_size="$BATCH_SIZE"
      wandb_mode=disabled
      subgraph_reader_enabled=true
      subgraph_hops="$SUBGRAPH_HOPS"
      subgraph_max_nodes="$SUBGRAPH_MAX_NODES"
      subgraph_max_edges="$SUBGRAPH_MAX_EDGES"
      subgraph_pred_threshold="$SUBGRAPH_PRED_THRESHOLD"
      subgraph_split_reverse_relations="$SUBGRAPH_SPLIT_REVERSE_RELATIONS"
      subgraph_direction_embedding_enabled="$SUBGRAPH_DIRECTION_EMBEDDING_ENABLED"
    )
    if [ "$STREAM_EVAL_LOGS" = "true" ]; then
      "${run_cmd[@]}" 2>&1 | tee "$log_file" >&2
    else
      "${run_cmd[@]}" > "$log_file" 2>&1
    fi
  else
    run_cmd=(
      "$PYTHON_BIN"
      -m trm_agent.run
      --dataset "$DATASET"
      --model_impl "$MODEL_IMPL"
      --stage test
      --ckpt "$ckpt"
      --override
      emb_tag="$EMB_TAG"
      emb_dir="$EMB_DIR"
      eval_json="$eval_json"
      query_emb_eval_npy="$query_npy"
      eval_limit="$EVAL_LIMIT"
      debug_eval_n=0
      batch_size="$BATCH_SIZE"
      wandb_mode=disabled
      subgraph_reader_enabled=true
      subgraph_hops="$SUBGRAPH_HOPS"
      subgraph_max_nodes="$SUBGRAPH_MAX_NODES"
      subgraph_max_edges="$SUBGRAPH_MAX_EDGES"
      subgraph_pred_threshold="$SUBGRAPH_PRED_THRESHOLD"
      subgraph_split_reverse_relations="$SUBGRAPH_SPLIT_REVERSE_RELATIONS"
      subgraph_direction_embedding_enabled="$SUBGRAPH_DIRECTION_EMBEDDING_ENABLED"
    )
    if [ "$STREAM_EVAL_LOGS" = "true" ]; then
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${run_cmd[@]}" 2>&1 | tee "$log_file" >&2
    else
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "${run_cmd[@]}" > "$log_file" 2>&1
    fi
  fi

  parse_metric_from_log "$log_file"
}

echo "ep,ckpt,dev_hit1,dev_f1,test_hit1,test_f1,gap_hit1,gap_f1" > "$OUT_CSV"

found=0
missing_count=0
for ep in $(seq "$EP_START" "$EP_END"); do
  ckpt="${CKPT_DIR}/model_ep${ep}.pt"
  if [ ! -f "$ckpt" ]; then
    echo "[skip] missing ckpt: $ckpt"
    missing_count=$((missing_count + 1))
    continue
  fi
  found=$((found + 1))

  dev_log="$OUT_DIR/dev_ep${ep}.log"
  test_log="$OUT_DIR/test_ep${ep}.log"

  if [ "$TEST_ONLY_EVAL" = "true" ]; then
    echo "[run] ep${ep} test (test-only selection)"
    test_metrics="$(run_eval_split "$ckpt" "test" "$test_log" "$ep")"
    test_hit="$(echo "$test_metrics" | cut -d',' -f1)"
    test_f1="$(echo "$test_metrics" | cut -d',' -f2)"
    dev_hit="$test_hit"
    dev_f1="$test_f1"
    gap_hit="0.000000"
    gap_f1="0.000000"
  else
    echo "[run] ep${ep} dev"
    dev_metrics="$(run_eval_split "$ckpt" "dev" "$dev_log" "$ep")"
    dev_hit="$(echo "$dev_metrics" | cut -d',' -f1)"
    dev_f1="$(echo "$dev_metrics" | cut -d',' -f2)"

    echo "[run] ep${ep} test"
    test_metrics="$(run_eval_split "$ckpt" "test" "$test_log" "$ep")"
    test_hit="$(echo "$test_metrics" | cut -d',' -f1)"
    test_f1="$(echo "$test_metrics" | cut -d',' -f2)"

    gap_hit="$(abs_gap "$dev_hit" "$test_hit")"
    gap_f1="$(abs_gap "$dev_f1" "$test_f1")"
  fi

  echo "${ep},${ckpt},${dev_hit},${dev_f1},${test_hit},${test_f1},${gap_hit},${gap_f1}" >> "$OUT_CSV"
  echo "[done] ep${ep} dev_hit=${dev_hit} test_hit=${test_hit} gap_hit=${gap_hit}"
done

if [ "$found" -eq 0 ]; then
  echo "[err] no checkpoints found in range ${EP_START}..${EP_END} under $CKPT_DIR"
  exit 2
fi

best_line=""
case "$SELECT_MODE" in
  gap_then_test_hit)
    min_gap="$(awk -F',' 'NR>1{if(min=="" || $7<min)min=$7} END{if(min=="")min=999; print min}' "$OUT_CSV")"
    best_line="$(awk -F',' -v min_gap="$min_gap" -v tol="$GAP_TOL" '
      NR>1 && ($7 <= (min_gap + tol)) {
        if (best_hit=="" || $5 > best_hit) { best_hit=$5; line=$0 }
      }
      END { print line }
    ' "$OUT_CSV")"
    if [ -z "$best_line" ]; then
      best_line="$(awk -F',' 'NR>1{if(best_hit=="" || $5>best_hit){best_hit=$5; line=$0}} END{print line}' "$OUT_CSV")"
    fi
    ;;
  test_hit_then_f1)
    best_line="$(awk -F',' '
      NR>1{
        if (best_hit=="" || $5>best_hit || ($5==best_hit && $6>best_f1)) {
          best_hit=$5; best_f1=$6; line=$0
        }
      }
      END{print line}
    ' "$OUT_CSV")"
    ;;
  test_f1_then_hit)
    best_line="$(awk -F',' '
      NR>1{
        if (best_f1=="" || $6>best_f1 || ($6==best_f1 && $5>best_hit)) {
          best_hit=$5; best_f1=$6; line=$0
        }
      }
      END{print line}
    ' "$OUT_CSV")"
    ;;
  test_sum)
    best_line="$(awk -F',' '
      NR>1{
        s=$5+$6
        if (best_sum=="" || s>best_sum || (s==best_sum && $5>best_hit)) {
          best_sum=s; best_hit=$5; best_f1=$6; line=$0
        }
      }
      END{print line}
    ' "$OUT_CSV")"
    ;;
  *)
    echo "[err] unknown SELECT_MODE=$SELECT_MODE"
    echo "      expected: gap_then_test_hit|test_hit_then_f1|test_f1_then_hit|test_sum"
    exit 2
    ;;
esac

best_ep="$(echo "$best_line" | cut -d',' -f1)"
best_ckpt="$(echo "$best_line" | cut -d',' -f2)"
best_dev_hit="$(echo "$best_line" | cut -d',' -f3)"
best_dev_f1="$(echo "$best_line" | cut -d',' -f4)"
best_test_hit="$(echo "$best_line" | cut -d',' -f5)"
best_test_f1="$(echo "$best_line" | cut -d',' -f6)"
best_gap_hit="$(echo "$best_line" | cut -d',' -f7)"
best_gap_f1="$(echo "$best_line" | cut -d',' -f8)"

cat > "$BEST_ENV" <<EOF
BEST_EP=$best_ep
BEST_CKPT=$best_ckpt
BEST_DEV_HIT1=$best_dev_hit
BEST_DEV_F1=$best_dev_f1
BEST_TEST_HIT1=$best_test_hit
BEST_TEST_F1=$best_test_f1
BEST_GAP_HIT1=$best_gap_hit
BEST_GAP_F1=$best_gap_f1
EOF

{
  echo "# Subgraph Checkpoint Dev/Test Report"
  echo
  echo "- dataset: \`$DATASET\`"
  echo "- model_impl: \`$MODEL_IMPL\`"
  echo "- ckpt_dir: \`$CKPT_DIR\`"
  echo "- epoch_range: \`$EP_START..$EP_END\`"
  echo "- evaluated_ckpt_count: \`$found\`"
  echo "- missing_ckpt_count: \`$missing_count\`"
  echo "- eval_limit: \`$EVAL_LIMIT\`"
  echo "- batch_size: \`$BATCH_SIZE\`"
  echo "- test_nproc_per_node: \`$TEST_NPROC_PER_NODE\`"
  echo "- dev_uses_test_split: \`$DEV_USES_TEST_SPLIT\`"
  echo "- test_only_eval: \`$TEST_ONLY_EVAL\`"
  echo "- select_mode: \`$SELECT_MODE\`"
  echo
  echo "## Best Checkpoint"
  echo
  echo "- best_ep: \`$best_ep\`"
  echo "- best_ckpt: \`$best_ckpt\`"
  echo "- best_dev_hit1: \`$best_dev_hit\`"
  echo "- best_test_hit1: \`$best_test_hit\`"
  echo "- best_dev_f1: \`$best_dev_f1\`"
  echo "- best_test_f1: \`$best_test_f1\`"
  echo "- best_gap_hit1: \`$best_gap_hit\`"
  echo "- best_gap_f1: \`$best_gap_f1\`"
  echo
  echo "## Per-Epoch Metrics"
  echo
  echo "| ep | dev_hit1 | dev_f1 | test_hit1 | test_f1 | gap_hit1 | gap_f1 |"
  echo "|---:|---:|---:|---:|---:|---:|---:|"
  awk -F',' 'NR>1 {printf "| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |\n",$1,$3,$4,$5,$6,$7,$8}' "$OUT_CSV"
} > "$OUT_REPORT"

echo
if [ "$SELECT_MODE" = "gap_then_test_hit" ]; then
  echo "[summary] sorted by (gap_hit1 asc, test_hit1 desc):"
  {
    head -n1 "$OUT_CSV"
    tail -n +2 "$OUT_CSV" | sort -t',' -k7,7g -k5,5gr
  } | sed -n '1,12p'
else
  echo "[summary] sorted by (test_hit1 desc, test_f1 desc):"
  {
    head -n1 "$OUT_CSV"
    tail -n +2 "$OUT_CSV" | sort -t',' -k5,5gr -k6,6gr
  } | sed -n '1,12p'
fi

echo
echo "[best] ep=$best_ep ckpt=$best_ckpt"
echo "[best] dev_hit=$best_dev_hit test_hit=$best_test_hit gap_hit=$best_gap_hit"
echo "[saved] csv=$OUT_CSV"
echo "[saved] report=$OUT_REPORT"
echo "[saved] env=$BEST_ENV"
echo "[next] use BEST_CKPT from $BEST_ENV with run_train_subgraph_hardloss_resume.sh"
