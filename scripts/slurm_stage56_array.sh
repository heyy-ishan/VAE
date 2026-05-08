#!/usr/bin/env bash
# Stage 5: parallel disentanglement eval — 5 models x 3 seeds = 15 tasks.
# Calls scripts/eval_run.py (proper checkpoint load + metric compute).
# Stage 6 aggregation auto-starts via slurm_stage56_agg.sh dependency.
#
# Usage: sbatch scripts/slurm_stage56_array.sh

#SBATCH --job-name=sc_vae_eval
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --array=0-14
#SBATCH --output=/scratch/ig2671/VAE/logs/slurm_%A_%a_eval.out
#SBATCH --error=/scratch/ig2671/VAE/logs/slurm_%A_%a_eval.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/ig2671/VAE}"
cd "$PROJECT_ROOT"
mkdir -p logs

module purge 2>/dev/null || true
module load anaconda3/2025.06 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true

export CONDA_ENVS_PATH=/scratch/ig2671/conda_envs
export CONDA_PKGS_DIRS=/scratch/ig2671/conda_pkgs
eval "$(conda shell.bash hook)"
conda activate sc_vae

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export H5_VAL="${NSYNTH_CACHE_VALID:-/scratch/ig2671/datasets/nsynth/cache/valid.h5}"

OUT_DIR="/scratch/ig2671/VAE/runs/full_eval_3seed/eval"
mkdir -p "${OUT_DIR}"

MODELS=(beta_vae beta_vae beta_vae beta_tcvae beta_tcvae beta_tcvae factor_vae factor_vae factor_vae ar_hvae ar_hvae ar_hvae sc_vae sc_vae sc_vae)
SEEDS=(0 1 2 0 1 2 0 1 2 0 1 2 0 1 2)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
OUT_JSON="${OUT_DIR}/${MODEL}_seed${SEED}.json"

echo "[eval] task=${SLURM_ARRAY_TASK_ID} model=${MODEL} seed=${SEED} node=$(hostname)"

if [[ -f "${OUT_JSON}" ]]; then
  echo "[skip] ${OUT_JSON} exists"
  exit 0
fi

# Find the version_0 run dir for this model+seed.
# Glob pattern matches the long Hydra override-dirname layout.
RUN_VERSION=$(find "${PROJECT_ROOT}/runs" -maxdepth 2 \
  -type d -name "version_*" \
  | grep "model=${MODEL}" \
  | grep "seed=${SEED}" \
  | sort | tail -1)

if [[ -z "${RUN_VERSION}" ]]; then
  echo "[error] no run dir found for model=${MODEL} seed=${SEED}" >&2
  exit 1
fi

echo "[eval] run_dir=${RUN_VERSION}"

python scripts/eval_run.py \
  --run_dir "${RUN_VERSION}" \
  --model   "${MODEL}" \
  --seed    "${SEED}" \
  --h5_val  "${H5_VAL}" \
  --out_json "${OUT_JSON}"

echo "[eval] done model=${MODEL} seed=${SEED}"
