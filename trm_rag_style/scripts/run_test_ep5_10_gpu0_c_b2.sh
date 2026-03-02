#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data2/workspace/heewon/KGQA"
PYTHON_BIN="/data2/workspace/heewon/anaconda3/envs/taiLab/bin/python"

CKPT_DIR="${CKPT_DIR:-${REPO_ROOT}/trm_agent/ckpt/cwq_trm_hier6_rearev_C_gpu123_b2_gatefusionhalt}"
EMB_DIR="${EMB_DIR:-${REPO_ROOT}/trm_agent/emb/cwq_e5_w4_g4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SPLIT="${SPLIT:-test}" # test | dev
CUDA_DEVICE="${CUDA_DEVICE:-0}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs/test_C_gpu0_ep5_10_b2}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

if [[ "${SPLIT}" == "dev" ]]; then
  EVAL_JSON="trm_agent/processed/cwq/dev.jsonl"
  QUERY_NPY="trm_agent/emb/cwq_e5_w4_g4/query_dev.npy"
elif [[ "${SPLIT}" == "test" ]]; then
  EVAL_JSON="trm_agent/processed/cwq/test.jsonl"
  QUERY_NPY="trm_agent/emb/cwq_e5_w4_g4/query_test.npy"
else
  echo "[err] SPLIT must be 'dev' or 'test' (got: ${SPLIT})"
  exit 2
fi

for ep in 5 6 7 8 9 10; do
  ckpt="${CKPT_DIR}/model_ep${ep}.pt"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[skip] missing ckpt: ${ckpt}"
    continue
  fi
  log_file="${LOG_DIR}/C_b2_ep${ep}_${SPLIT}.log"
  echo "[run] C(b2) ep=${ep} split=${SPLIT} ckpt=${ckpt}"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
  "${PYTHON_BIN}" -m trm_agent.run \
    --dataset cwq \
    --model_impl trm_hier6 \
    --stage test \
    --ckpt "${ckpt}" \
    --override \
      subgraph_reader_enabled=true \
      emb_tag=e5_w4_g4 \
      emb_dir=trm_agent/emb/cwq_e5_w4_g4 \
      eval_json="${EVAL_JSON}" \
      query_emb_eval_npy="${QUERY_NPY}" \
      batch_size="${BATCH_SIZE}" \
      eval_limit=-1 \
      wandb_mode=offline \
      subgraph_loss_mode=rearev_kl \
      subgraph_add_reverse_edges=true \
      subgraph_split_reverse_relations=true \
      subgraph_direction_embedding_enabled=true \
      subgraph_rearev_latent_reasoning_enabled=true \
      subgraph_rearev_latent_residual_alpha=0.25 \
      subgraph_rearev_global_gate_enabled=true \
      subgraph_rearev_logit_global_fusion_enabled=true \
      subgraph_rearev_dynamic_halting_enabled=true \
      subgraph_rearev_dynamic_halting_threshold=0.90 \
      subgraph_rearev_dynamic_halting_min_steps=1 \
    2>&1 | tee "${log_file}"
done

echo "[done] C(b2) ep5~10 ${SPLIT} evaluation finished"
