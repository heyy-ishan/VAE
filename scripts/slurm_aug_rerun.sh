#!/usr/bin/env bash
# Rerun 2 failed aug ablations (tasks 13-14 from stage3 array).
#
# Usage: sbatch scripts/slurm_aug_rerun.sh

#SBATCH --job-name=sc_vae_aug
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/ig2671/VAE/logs/slurm_%A_%a_aug_rerun.out
#SBATCH --error=/scratch/ig2671/VAE/logs/slurm_%A_%a_aug_rerun.err

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
export NSYNTH_CACHE_TRAIN="${NSYNTH_CACHE_TRAIN:-/scratch/ig2671/datasets/nsynth/cache/train.h5}"
export NSYNTH_CACHE_VALID="${NSYNTH_CACHE_VALID:-/scratch/ig2671/datasets/nsynth/cache/valid.h5}"
export MOISESDB_ROOT="${MOISESDB_ROOT:-/scratch/ig2671/datasets/moisesdb}"
export CREPE_CACHE="${CREPE_CACHE:-/scratch/ig2671/datasets/crepe_cache.h5}"

ABLATION_IDS=(aug_uniform_half aug_discrete)
ABLATION_OVERRIDES=(+lit.g_cents_dist=uniform_half +lit.g_cents_dist=discrete_semitones)

ID=${ABLATION_IDS[$SLURM_ARRAY_TASK_ID]}
OVERRIDES=${ABLATION_OVERRIDES[$SLURM_ARRAY_TASK_ID]}

echo "[aug_rerun] task=${SLURM_ARRAY_TASK_ID} ablation=${ID} overrides=${OVERRIDES} node=$(hostname)"

set -f
python -m src.training.cli \
  --config-name base \
  model=sc_vae \
  data=nsynth_bass \
  seed=0 \
  trainer.max_epochs=100 \
  +trainer.precision=bf16-mixed \
  +lit.beta_schedule=cyclical \
  +lit.n_cycles=4 \
  +lit.beta_peak=4.0 \
  "hydra.run.dir=/scratch/ig2671/VAE/runs/ablations/${ID}_seed0" \
  ${OVERRIDES}
set +f

echo "[aug_rerun] done ablation=${ID}"
