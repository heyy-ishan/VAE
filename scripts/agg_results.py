#!/usr/bin/env python3
"""Aggregate eval JSONs → results.tex + figures.

eval_run.py writes flat JSON (mig, sap, etc. at top level).
report.py expects {"model": ..., "seed": ..., "metrics": {...}}.
This script reshapes and calls report functions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.report import (
    write_latex_table,
    write_dci_heatmap,
    write_pareto_scatter,
)

EVAL_DIR = Path("/scratch/ig2671/VAE/runs/full_eval_3seed/eval")
FIG_DIR = Path("/scratch/ig2671/VAE/docs/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

METRIC_KEYS = [
    "mig", "sap", "modularity", "factor_vae_score",
    "dci_d", "dci_c", "dci_i",
]
MODEL_ORDER = ["beta_vae", "beta_tcvae", "factor_vae", "ar_hvae", "sc_vae"]
METRIC_ORDER = METRIC_KEYS  # same order for table

raw_runs = [json.loads(p.read_text()) for p in sorted(EVAL_DIR.glob("*_seed*.json"))]
if not raw_runs:
    raise SystemExit(f"[agg] no JSON files found in {EVAL_DIR}")

print(f"[agg] loaded {len(raw_runs)} runs: {[r['model']+'_s'+str(r['seed']) for r in raw_runs]}")

# Reshape: move metric scalars into r["metrics"] sub-dict
runs = []
for r in raw_runs:
    metrics = {k: v for k, v in r.items()
               if k not in ("model", "seed", "dci_importance", "factor_names")}
    runs.append({
        "model": r["model"],
        "seed": r["seed"],
        "metrics": metrics,
        "dci_importance": r.get("dci_importance"),
        "factor_names": r.get("factor_names"),
    })

# LaTeX table
out_tex = EVAL_DIR / "results.tex"
write_latex_table(
    runs=runs,
    out_path=out_tex,
    metric_order=METRIC_ORDER,
    model_order=MODEL_ORDER,
)
print(f"[agg] wrote {out_tex}")

# DCI heatmaps (one per model, first seed that has it)
seen: set[str] = set()
for r in runs:
    model = r["model"]
    if model in seen or r["dci_importance"] is None:
        continue
    R = np.asarray(r["dci_importance"])
    fnames = r["factor_names"] or [f"v{i}" for i in range(R.shape[1])]
    out_pdf = FIG_DIR / f"dci_heatmap_{model}.pdf"
    write_dci_heatmap(
        importance_matrix=R,
        factor_names=fnames,
        out_path=out_pdf,
        title=model,
    )
    print(f"[agg] wrote {out_pdf}")
    seen.add(model)

# Pareto scatter: MIG vs SAP
out_pareto = FIG_DIR / "pareto_mig_sap.pdf"
write_pareto_scatter(
    runs=runs,
    x_metric="mig",
    y_metric="sap",
    out_path=out_pareto,
)
print(f"[agg] wrote {out_pareto}")

print("[agg] done")
