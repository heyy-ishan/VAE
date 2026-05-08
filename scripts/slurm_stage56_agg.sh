#!/usr/bin/env bash
# Stage 6: metric rollup — runs after slurm_stage56_array.sh completes.
# Submit with: sbatch --dependency=afterok:<ARRAY_JOB_ID> scripts/slurm_stage56_agg.sh
#
# Usage: sbatch --dependency=afterok:JOBID scripts/slurm_stage56_agg.sh

#SBATCH --job-name=sc_vae_agg
#SBATCH --account=torch_pr_932_general
#SBATCH --partition=h200_public
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/ig2671/VAE/logs/slurm_%j_agg.out
#SBATCH --error=/scratch/ig2671/VAE/logs/slurm_%j_agg.err

set -eo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/ig2671/VAE}"
cd "$PROJECT_ROOT"

module purge 2>/dev/null || true
module load anaconda3/2025.06 2>/dev/null || true

export CONDA_ENVS_PATH=/scratch/ig2671/conda_envs
export CONDA_PKGS_DIRS=/scratch/ig2671/conda_pkgs
eval "$(conda shell.bash hook)"
conda activate sc_vae

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

OUT_DIR="/scratch/ig2671/VAE/runs/full_eval_3seed/eval"
FIG_DIR="${PROJECT_ROOT}/docs/figures"

echo "[agg] aggregating results from ${OUT_DIR}"

python - <<PY
from pathlib import Path
import json
import numpy as np
from src.evaluation.report import (
    write_latex_table, write_dci_heatmap, write_pareto_scatter,
)

out_dir = Path("${OUT_DIR}")
fig_dir = Path("${FIG_DIR}")
fig_dir.mkdir(parents=True, exist_ok=True)

runs = [json.loads(p.read_text()) for p in sorted(out_dir.glob("*_seed*.json"))]
if not runs:
    raise SystemExit("[agg] no JSON files found")

metric_order = [
    "mig", "sap", "modularity", "factor_vae_score",
    "dci_d", "dci_c", "dci_i",
    "srr", "invariance_ratio", "equivariance_ratio", "pitch_acc_50",
]
model_order = ["beta_vae", "beta_tcvae", "factor_vae", "ar_hvae", "sc_vae"]

write_latex_table(
    runs=runs,
    out_path=out_dir / "results.tex",
    metric_order=metric_order,
    model_order=model_order,
)

seen: set[str] = set()
for r in runs:
    model = r["model"]
    if model in seen or "dci_importance" not in r:
        continue
    R = np.asarray(r["dci_importance"])
    write_dci_heatmap(
        importance_matrix=R,
        factor_names=r.get("factor_names", [f"v{i}" for i in range(R.shape[1])]),
        out_path=fig_dir / f"dci_heatmap_{model}.pdf",
        title=model,
    )
    seen.add(model)

write_pareto_scatter(
    runs=runs, x_metric="srr", y_metric="sap",
    out_path=fig_dir / "pareto_srr_sap.pdf",
)
print(f"[ok] results.tex + figures written")
PY

echo "[agg] done"
