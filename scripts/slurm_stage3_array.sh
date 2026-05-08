#!/usr/bin/env bash
# Stage 3: 15 ablations of sc_vae, seed=0, 100 epochs each, all parallel.
#
# Usage: sbatch scripts/slurm_stage3_array.sh

#SBATCH --job-name=sc_vae_s3
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-14
#SBATCH --output=/scratch/ig2671/VAE/logs/slurm_%A_%a_stage3.out
#SBATCH --error=/scratch/ig2671/VAE/logs/slurm_%A_%a_stage3.err

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

# Ablation id and extra overrides (must match run_ablations.sh ABLATIONS_READY order)
ABLATION_IDS=(
  full
  no_l_inv
  no_l_equi
  no_l_swap
  rep_identity
  rep_translation
  dims_swapped
  dims_unswapped
  beta_const
  beta_cyclical
  beta_asym_high_c
  beta_asym_high_s
  unpaired_batch
  aug_uniform_half
  aug_discrete
)
ABLATION_OVERRIDES=(
  ""
  "lit.lambda_inv=0"
  "lit.lambda_equi=0"
  "lit.lambda_swap=0"
  "model.rep_type=identity"
  "model.rep_type=translation"
  "model.d_s=16 model.d_c=64"
  "model.d_s=64 model.d_c=16"
  "lit.beta_schedule=constant lit.beta_s=4 lit.beta_c=4"
  "lit.beta_schedule=cyclical lit.beta_s=4 lit.beta_c=4"
  "lit.beta_s=1 lit.beta_c=8"
  "lit.beta_s=8 lit.beta_c=1"
  "+data.paired=false"
  "+lit.g_cents_dist=uniform_half"
  "+lit.g_cents_dist=discrete_semitones"
)

ID=${ABLATION_IDS[$SLURM_ARRAY_TASK_ID]}
OVERRIDES=${ABLATION_OVERRIDES[$SLURM_ARRAY_TASK_ID]}

echo "[stage3] task=${SLURM_ARRAY_TASK_ID} ablation=${ID} overrides='${OVERRIDES}' node=$(hostname)"

# shellcheck disable=SC2086
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

echo "[stage3] done ablation=${ID}"
