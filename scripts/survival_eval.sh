#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 <os|lffs|rffs|dffs> <model.ckpt> <train.csv> <test.csv> <nifti_dir> [gpu_id]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TASK="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
CKPT_PATH="$(realpath "$2")"
TRAIN_CSV="$(realpath "$3")"
TEST_CSV="$(realpath "$4")"
NIFTI_DIR="$(realpath "$5")"
GPU_ID="${6:-${GPU_ID:-0}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/survival_${TASK}}"

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
  --config-name test \
  model=radcure_model_img_vit \
  datamodule=radcure_train_test \
  "ckpt_path=${CKPT_PATH}" \
  "datamodule.train_filename=${TRAIN_CSV}" \
  "datamodule.test_filename=${TEST_CSV}" \
  "datamodule.train_cache_dir=${NIFTI_DIR}" \
  "datamodule.test_cache_dir=${NIFTI_DIR}" \
  "datamodule.batch_size=${BATCH_SIZE:-10}" \
  "datamodule.num_workers=${NUM_WORKERS:-8}" \
  model.patch_xy=80 model.patch_z=48 \
  model.n_dense=2 model.dense_factor=1 model.time_bins=8 model.dropout=0.25 \
  postprocessor.enable=false \
  "output_dir=${OUTPUT_DIR}/results" \
  "logger.csv.save_dir=${OUTPUT_DIR}" \
  "hydra.run.dir=${OUTPUT_DIR}" \
  hydra.job.chdir=true
