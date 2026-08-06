#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 <os|lffs|rffs|dffs> <train.csv> <id_test.csv> <nifti_dir> [gpu_id]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TASK="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
TRAIN_CSV="$(realpath "$2")"
TEST_CSV="$(realpath "$3")"
NIFTI_DIR="$(realpath "$4")"
GPU_ID="${5:-${GPU_ID:-0}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/train_${TASK}}"

case "${TASK}" in
  os|lffs|rffs|dffs) MODEL_CONFIG="radcure_model_img_vit_${TASK}" ;;
  *) echo "Unsupported task: ${TASK}" >&2; exit 2 ;;
esac

mkdir -p "${OUTPUT_DIR}" "${REPO_ROOT}/outputs/matplotlib"
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
PYTHONPATH="${REPO_ROOT}/torchmtlr:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
MPLCONFIGDIR="${REPO_ROOT}/outputs/matplotlib" \
HYDRA_FULL_ERROR=1 \
"${PYTHON_BIN}" train.py \
  "model=${MODEL_CONFIG}" \
  datamodule=radcure_train_test \
  "datamodule.train_filename=${TRAIN_CSV}" \
  "datamodule.test_filename=${TEST_CSV}" \
  "datamodule.train_cache_dir=${NIFTI_DIR}" \
  "datamodule.test_cache_dir=${NIFTI_DIR}" \
  "datamodule.Fold=${FOLD:-5}" \
  "datamodule.batch_size=${BATCH_SIZE:-32}" \
  "datamodule.num_workers=${NUM_WORKERS:-8}" \
  model.patch_xy=80 model.patch_z=48 \
  model.n_dense=2 model.dense_factor=1 model.time_bins=8 \
  "model.lr=${LR:-1e-3}" model.dropout=0.25 \
  "trainer.max_epochs=${MAX_EPOCHS:-400}" \
  "name=RadCure_ImgVit_${TASK}" \
  "callbacks.model_checkpoint.dirpath=${OUTPUT_DIR}/checkpoints" \
  "logger.csv.save_dir=${OUTPUT_DIR}" \
  "hydra.run.dir=${OUTPUT_DIR}" \
  hydra.job.chdir=true
