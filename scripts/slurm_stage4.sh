#!/usr/bin/env bash
# Stage 4: MoisesDB mini-arm, sc_vae, seed=0, 50 epochs.
#
# Usage: sbatch scripts/slurm_stage4.sh

#SBATCH --job-name=sc_vae_s4
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/scratch/ig2671/VAE/logs/slurm_%j_stage4.out
#SBATCH --error=/scratch/ig2671/VAE/logs/slurm_%j_stage4.err

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

echo "[stage4] MoisesDB node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python -m src.training.cli \
  model=sc_vae \
  data=moisesdb_bass \
  seed=0 \
  trainer.max_epochs=50 \
  +trainer.precision=bf16-mixed \
  +lit.beta_schedule=cyclical \
  +lit.n_cycles=4 \
  +lit.beta_peak=4.0 \
  "hydra.run.dir=/scratch/ig2671/VAE/runs/stage4/moisesdb_bass_seed0"

echo "[stage4] done"
