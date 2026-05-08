#!/usr/bin/env bash
# Stage 2: 9 runs (sc_vae + beta_vae + beta_tcvae) x seeds {0,1,2}, 100 epochs.
# Each array task = one model+seed combo, runs in parallel on its own GPU.
#
# Usage: sbatch scripts/slurm_stage2_array.sh

#SBATCH --job-name=sc_vae_s2
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-8
#SBATCH --output=/scratch/ig2671/VAE/logs/slurm_%A_%a_stage2.out
#SBATCH --error=/scratch/ig2671/VAE/logs/slurm_%A_%a_stage2.err

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

# Map array task index → model + seed
MODELS=(sc_vae sc_vae sc_vae beta_vae beta_vae beta_vae beta_tcvae beta_tcvae beta_tcvae)
SEEDS=(0 1 2 0 1 2 0 1 2)
BETA_S=(1.0 1.0 1.0 4.0 4.0 4.0 6.0 6.0 6.0)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
BS=${BETA_S[$SLURM_ARRAY_TASK_ID]}

echo "[stage2] task=${SLURM_ARRAY_TASK_ID} model=${MODEL} seed=${SEED} beta_s=${BS} node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python -m src.training.cli \
  model="${MODEL}" \
  data=nsynth_bass \
  seed="${SEED}" \
  trainer.max_epochs=100 \
  +trainer.precision=bf16-mixed \
  lit.beta_s="${BS}" \
  +lit.beta_schedule=cyclical \
  +lit.n_cycles=4 \
  +lit.beta_peak=4.0 \
  "hydra.run.dir=/scratch/ig2671/VAE/runs/stage2/${MODEL}_seed${SEED}"

echo "[stage2] done task=${SLURM_ARRAY_TASK_ID}"
