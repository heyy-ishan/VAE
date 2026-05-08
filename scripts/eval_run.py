#!/usr/bin/env python3
"""Standalone eval: load checkpoint, encode val set, compute disentanglement metrics.

Usage:
    python scripts/eval_run.py \
        --run_dir /scratch/ig2671/VAE/runs/<long_dirname>/version_0 \
        --model sc_vae --seed 0 \
        --h5_val /scratch/ig2671/datasets/nsynth/cache/valid.h5 \
        --out_json /path/to/output.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import hydra

from src.data.nsynth_cached import NSynthBassCached
from src.training.lit_module import VAELitModule
from src.evaluation._common import extract_latents
from src.evaluation import mig as mig_mod
from src.evaluation import dci as dci_mod
from src.evaluation import sap as sap_mod
from src.evaluation import modularity as mod_mod
from src.evaluation import factor_vae_score as fvae_mod
from src.utils.seed import set_seeds


FACTOR_KEYS = ["pitch", "velocity", "instrument_source", "instrument_id"]
FACTOR_TYPES = {
    "pitch": "continuous",
    "velocity": "continuous",
    "instrument_source": "discrete",
    "instrument_id": "discrete",
}
LATENT_KEYS = ["mu_s", "mu_c", "mu"]


def _collate(batch: list[dict]) -> dict:
    out: dict = {}
    for k in batch[0]:
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], dict):
            out[k] = _collate(vals)
        else:
            out[k] = torch.as_tensor(vals)
    return out


@torch.no_grad()
def encode_val_set(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model.train(False)
    z_list: list[np.ndarray] = []
    factors_list: dict[str, list[np.ndarray]] = {k: [] for k in FACTOR_KEYS}

    for batch in loader:
        x = batch["x"].to(device)
        raw_out = model.forward(x)
        # AR_HVAE returns lists for mu/logvar/z — concatenate to single tensors.
        out = {
            k: torch.cat(v, dim=1) if isinstance(v, list) else v
            for k, v in raw_out.items()
        }
        z = extract_latents(out, LATENT_KEYS)
        z_list.append(z.cpu().numpy())

        labels = batch.get("labels", {})
        for k in FACTOR_KEYS:
            if k in labels:
                factors_list[k].append(labels[k].numpy())

    z_np = np.concatenate(z_list, axis=0)
    factors_np = {
        k: np.concatenate(v, axis=0)
        for k, v in factors_list.items()
        if v
    }
    return z_np, factors_np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True,
                        help="Path to version_N dir (contains checkpoints/ subdir)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--h5_val", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_seeds(args.seed)

    run_dir = Path(args.run_dir)

    # .hydra/config.yaml may be 1-3 levels up depending on layout.
    # CSVLogger: save_dir/name/version_N → .hydra is next to name/ dir.
    hydra_cfg_candidates = [
        run_dir.parent.parent / ".hydra" / "config.yaml",
        run_dir.parent / ".hydra" / "config.yaml",
        run_dir / ".hydra" / "config.yaml",
    ]
    hydra_cfg_path = next((p for p in hydra_cfg_candidates if p.exists()), None)

    if hydra_cfg_path is not None:
        cfg = OmegaConf.load(hydra_cfg_path)
        model_inst = hydra.utils.instantiate(cfg.model)
    else:
        # Fallback: run_manifest.json saved by training CLI contains full config dict.
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"No .hydra/config.yaml or run_manifest.json found in {run_dir}. "
                f"Tried hydra candidates: {[str(p) for p in hydra_cfg_candidates]}"
            )
        manifest = json.loads(manifest_path.read_text())
        cfg = OmegaConf.create(manifest["config"])
        model_inst = hydra.utils.instantiate(cfg.model)

    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {ckpt_dir}")
    ckpt_path = ckpts[-1]
    print(f"[eval] loading {ckpt_path}")

    lit = VAELitModule.load_from_checkpoint(
        str(ckpt_path), model=model_inst, map_location=device
    )
    lit = lit.to(device)
    lit.train(False)

    val_ds = NSynthBassCached(args.h5_val)
    val_loader = DataLoader(
        val_ds, batch_size=128, num_workers=4, shuffle=False,
        collate_fn=_collate, pin_memory=(device.type == "cuda"),
    )

    print(f"[eval] encoding {len(val_ds)} val samples")
    z_np, factors_np = encode_val_set(lit.model, val_loader, device)
    print(f"[eval] z={z_np.shape}  factors={list(factors_np.keys())}")

    results: dict = {"model": args.model, "seed": args.seed}

    print("[eval] MIG")
    r = mig_mod.compute_mig_from_arrays(z=z_np, factors=factors_np, n_bins=20)
    results["mig"] = float(r["mig"])

    print("[eval] DCI")
    r = dci_mod.compute_dci_from_arrays(
        z=z_np, factors=factors_np,
        factor_types=FACTOR_TYPES,
        gbt_kwargs={"n_estimators": 500, "max_depth": 6},
        test_size=0.2,
    )
    results["dci_d"] = float(r["disentanglement"])
    results["dci_c"] = float(r["completeness"])
    results["dci_i"] = float(r["informativeness"])
    results["dci_importance"] = np.asarray(r["importance_matrix"]).tolist()
    results["factor_names"] = list(r["factor_names"])

    print("[eval] SAP")
    r = sap_mod.compute_sap_from_arrays(
        z=z_np, factors=factors_np,
        factor_types=FACTOR_TYPES,
        svc_kwargs={"C": 1.0},
        test_size=0.2,
    )
    results["sap"] = float(r["sap"])

    print("[eval] Modularity")
    r = mod_mod.compute_modularity_from_arrays(
        z=z_np, factors=factors_np, n_bins=20, min_theta_max=0.05
    )
    results["modularity"] = float(r["modularity"])

    print("[eval] FactorVAE score")
    r = fvae_mod.compute_factor_vae_from_arrays(
        z=z_np, factors=factors_np,
        n_votes=800, batch_size=64, test_size=0.2,
    )
    results["factor_vae_score"] = float(r["factor_vae_score"])

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
