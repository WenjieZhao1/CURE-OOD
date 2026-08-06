#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 8 ]]; then
  echo "Usage: $0 <os|lffs|rffs|dffs> <model.ckpt> <train.csv> <id.csv> <ood.csv> <nifti_dir> [method] [gpu_id]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TASK="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
CKPT_PATH="$(realpath "$2")"
TRAIN_CSV="$(realpath "$3")"
ID_CSV="$(realpath "$4")"
OOD_CSV="$(realpath "$5")"
NIFTI_DIR="$(realpath "$6")"
METHOD="${7:-msp}"
GPU_ID="${8:-${GPU_ID:-0}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/ood_${TASK}_${METHOD}}"

case "${TASK}" in
  os) FOCUS_TASK="OS" ;;
  lffs) FOCUS_TASK="LFFS" ;;
  rffs) FOCUS_TASK="RFFS" ;;
  dffs) FOCUS_TASK="DFFS" ;;
  *) echo "Unsupported task: ${TASK}" >&2; exit 2 ;;
esac

mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/outputs/matplotlib"
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONPATH="${REPO_ROOT}/torchmtlr:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
MPLCONFIGDIR="${REPO_ROOT}/outputs/matplotlib" \
HYDRA_FULL_ERROR=1 \
FOCUS_TASK="${FOCUS_TASK}" \
"${PYTHON_BIN}" test.py \
  --config-name test_mixed \
  model=radcure_model_img_vit \
  datamodule=radcure_mixed_test \
  "ckpt_path=${CKPT_PATH}" \
  "datamodule.train_filename=${TRAIN_CSV}" \
  "datamodule.id_data_path=${ID_CSV}" \
  "datamodule.ood_data_path=${OOD_CSV}" \
  "datamodule.train_cache_dir=${NIFTI_DIR}" \
  "datamodule.test_cache_dir=${NIFTI_DIR}" \
  "datamodule.batch_size=${BATCH_SIZE:-10}" \
  "datamodule.num_workers=${NUM_WORKERS:-8}" \
  datamodule.ensure_balanced_batches=false datamodule.shuffle=false \
  model.patch_xy=80 model.patch_z=48 \
  model.n_dense=2 model.dense_factor=1 model.time_bins=8 model.dropout=0.25 \
  hazard_reference.mode=none \
  model.hazard_reference_mode=none \
  postprocessor.enable=true \
  "postprocessor.name=${METHOD}" \
  postprocessor.evaluate_per_task=true \
  "postprocessor.metrics_output=${OUTPUT_DIR}/${METHOD}_metrics.csv" \
  "output_dir=${OUTPUT_DIR}/results" \
  "logger.csv.save_dir=${OUTPUT_DIR}" \
  "hydra.run.dir=${OUTPUT_DIR}" \
  hydra.job.chdir=true
