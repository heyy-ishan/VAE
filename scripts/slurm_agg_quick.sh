#!/usr/bin/env bash
#SBATCH --job-name=sc_vae_agg_quick
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:15:00
#SBATCH --output=/scratch/ig2671/VAE/logs/slurm_%j_agg_quick.out
#SBATCH --error=/scratch/ig2671/VAE/logs/slurm_%j_agg_quick.err

set -eo pipefail
cd /scratch/ig2671/VAE

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh 2>/dev/null || true
conda activate /scratch/ig2671/conda_envs/sc_vae

export PYTHONPATH="/scratch/ig2671/VAE${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/agg_results.py
