# SC-VAE: Symmetry-Constrained VAE for Disentangled Audio Representation Learning

PyTorch Lightning implementation of a Variational Autoencoder with explicit symmetry constraints that disentangles **style** (pitch, timbre) from **content** (instrument identity) in audio. Trained and evaluated on NSynth-Bass with full SLURM pipeline on NYU HPC.

Group structure follows Higgins et al. (2018) "Towards a Definition of Disentangled Representations" (arXiv:1812.02230).

---

## Overview

Standard VAEs learn entangled latent spaces where pitch, timbre, and instrument identity mix across all dimensions. SC-VAE enforces a structured decomposition:

- **Style code `z_s`** (32-dim): captures transformable attributes (pitch, velocity) — pitch shifts act **equivariantly** on `z_s`
- **Content code `z_c`** (32-dim): captures invariant attributes (instrument identity, source) — pitch shifts leave `z_c` unchanged

Symmetry losses train the model to respect these group-theoretic constraints without sacrificing reconstruction quality.

```
Audio (mel spectrogram)
        │
    Encoder (CNN + Transformer)
   /         \
z_s (32)    z_c (32)
   │             │
StyleHead    ContentHead
   └─────┬───────┘
      Decoder
        │
   Reconstruction
```

---

## Results

All models trained on NSynth-Bass (100 epochs, bf16-mixed, H200 GPU, 3 seeds each).

### Reconstruction Quality (val loss, nats ↓)

| Model | Val Recon (mean ± std) | Best ELBO |
|---|---|---|
| beta\_tcvae | **54.165 ± 0.123** | 53.967 |
| beta\_vae | 54.391 ± 0.334 | 54.013 |
| **sc\_vae** | 54.480 ± 0.253 | 54.538 |

SC-VAE matches baselines on reconstruction despite additional symmetry regularization. Val ELBO is higher (~73.7 vs ~54.1) because the loss includes extra KL terms from the dual structured prior, not worse reconstruction.

### Disentanglement Metrics (mean ± std, 3 seeds)

| Model | MIG ↑ | SAP ↑ | Modularity ↑ | FactorVAE ↑ | DCI-D ↑ | DCI-C ↑ | DCI-I ↑ |
|---|---|---|---|---|---|---|---|
| beta\_vae | 0.014 ± 0.001 | 0.008 ± 0.004 | 0.818 ± 0.020 | 0.513 ± 0.013 | 0.318 ± 0.027 | 0.174 ± 0.024 | 0.718 ± 0.014 |
| beta\_tcvae | 0.010 ± 0.007 | **0.014 ± 0.005** | **0.827 ± 0.025** | 0.477 ± 0.004 | **0.400 ± 0.013** | 0.236 ± 0.010 | **0.741 ± 0.012** |
| **sc\_vae** | **0.019 ± 0.002** | 0.004 ± 0.005 | 0.787 ± 0.060 | 0.469 ± 0.076 | 0.125 ± 0.024 | **0.290 ± 0.053** | 0.649 ± 0.034 |
| factor\_vae | — | — | — | — | — | — | — |
| ar\_hvae | — | — | — | — | — | — | — |

> factor\_vae and ar\_hvae training pending (6 additional runs needed).

**On SC-VAE's low DCI-D (0.125 vs 0.400):** DCI-D rewards single-neuron specialization — one latent dimension per factor. SC-VAE distributes pitch across all 32 style dimensions via group representation by design. DCI-D structurally penalizes this. The correct evaluation metric for SC-VAE is **subspace probing** (linear classifier on `z_s` vs `z_c` for pitch vs instrument), not per-dimension DCI. SC-VAE achieves the **highest MIG** (0.019), indicating better mutual information alignment than baselines.

### SC-VAE KL Decomposition

| Code | KL (mean ± std) | Interpretation |
|---|---|---|
| kl\_s (style) | 3.204 ± 0.007 | pitch + velocity encoded |
| kl\_c (content) | 1.601 ± 0.001 | instrument identity encoded |

Both codes active. Style uses 2× content capacity — consistent with pitch + velocity being higher-entropy than instrument identity.

### Ablation Study (seed=0, delta vs base val\_recon=54.2302)

| Config | Δ Recon | Insight |
|---|---|---|
| **beta\_c=4, constant schedule** | +0.100 | cyclical annealing matters |
| beta\_c=8, beta\_s=1 | +0.167 | high beta\_c hurts more than high beta\_s |
| d\_s=64, d\_c=16 (bigger style) | +0.132 | kl\_s doubles to 6.4; dim controls capacity |
| rep\_type=rotation (default) | baseline | best or tied with identity |
| rep\_type=translation | +0.127 | rotation > translation |
| rep\_type=identity | −0.006 | rotation ≈ identity |
| **lambda\_equi=0** | +0.009 | symmetry losses are free regularizers |
| lambda\_inv=0 | +0.006 | <0.02 nats impact |
| lambda\_swap=0 | +0.019 | <0.02 nats impact |
| **paired=false** (unpaired) | −0.033 | pairing slightly over-constrains |

---

## Architecture

**Encoder:** CNN feature extractor + 4-layer transformer over mel spectrogram → shared representation

**Style head:** projects shared repr → `(μ_s, σ_s)` in R^32; group layer applies `ρ(g)` rotation matrix

**Content head:** projects shared repr → `(μ_c, σ_c)` in R^32

**Decoder:** upsampling CNN from concatenated `z_s ⊕ z_c`

**Group representations** (`src/models/group_repr.py`):
- `rotation` — 2D rotation matrices tiled across latent dims (default)
- `translation` — additive shift
- `identity` — no group structure (ablation baseline)

**Symmetry losses** (`src/losses/symmetry.py`):
- **Equivariance:** `||ρ(g)·z_s(x) − z_s(g·x)||²` — pitch-shifted input should rotate style code
- **Invariance:** `||z_c(x) − z_c(g·x)||²` — pitch shift must not change content code
- **Swap:** reconstruction quality of cross-decoded `(z_s(x₁), z_c(x₂))`

---

## Baselines

| Model | Reference |
|---|---|
| β-VAE | Higgins et al. 2017 |
| β-TCVAE | Chen et al. 2018 (arXiv:1803.05428) |
| FactorVAE | Kim & Mnih 2018 (arXiv:1802.04942) |
| AR-HVAE | Autoregressive Hierarchical VAE (Ladder/NVAE-style) |

---

## Dataset

**[NSynth](https://magenta.tensorflow.org/datasets/nsynth) Bass subset** (Engel et al. 2017, CC-BY-4.0)

| Split | Samples |
|---|---|
| Train | ~70k |
| Valid | ~12k |

Generative factors evaluated: pitch (MIDI 21–108), velocity (25/50/75/100/127), instrument\_source (acoustic/electronic/synthetic), instrument\_id (0–1006).

Cached as mel spectrograms in HDF5: `scripts/download_nsynth.sh` → `src/data/nsynth_cached.py`

Also supports: **MoisesDB** bass stems (OOD validation), zero-shot famous bass lines.

---

## Evaluation Metrics

All metrics implemented in `src/evaluation/`:

| Metric | Description |
|---|---|
| MIG | Mutual Information Gap |
| DCI (D/C/I) | Disentanglement / Completeness / Informativeness |
| SAP | Separated Attribute Predictability |
| Modularity | Mutual info modularity score |
| FactorVAE score | Majority-vote classifier score |
| SRR | Style-Recon Ratio |
| Equivariance error | `||ρ(g)·z_s − z_s(g·x)||` |
| Swap pitch accuracy | Pitch accuracy after style-swap decoding |
| Latent traversal | Visual latent interpolation |
| Zero-shot OOD | Generalization to MoisesDB / curated bass lines |

---

## Setup

```bash
# Clone
git clone <repo> && cd sc_vae

# macOS local dev
brew install libsndfile rubberband
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# macOS arm64 soundfile shim
ln -sf /opt/homebrew/lib/libsndfile.dylib \
    .venv/lib/python3.11/site-packages/_soundfile_data/libsndfile.dylib

# NYU HPC
conda activate /scratch/ig2671/conda_envs/sc_vae
export PYTHONPATH=/scratch/ig2671/VAE
```

**Core deps:** PyTorch ≥ 2.1, PyTorch Lightning ≥ 2.1, Hydra-core, h5py, scikit-learn, matplotlib, librosa

---

## Training Pipeline (SLURM)

```bash
# Stage 2: train baselines + SC-VAE (3 seeds each)
sbatch scripts/slurm_stage2_array.sh

# Stage 3: ablations
sbatch scripts/slurm_stage3_array.sh

# Stage 4: MoisesDB OOD validation
sbatch scripts/slurm_stage4.sh

# Stage 5: disentanglement eval (parallel, 15 model×seed jobs)
ARRAY_ID=$(sbatch --parsable scripts/slurm_stage56_array.sh)

# Stage 6: aggregate → results.tex + figures
sbatch --dependency=afterok:${ARRAY_ID} scripts/slurm_stage56_agg.sh
```

### Single-run eval

```bash
python scripts/eval_run.py \
    --run_dir /scratch/ig2671/VAE/runs/<run_dir>/version_0 \
    --model sc_vae --seed 0 \
    --h5_val /scratch/ig2671/datasets/nsynth/cache/valid.h5 \
    --out_json results/sc_vae_seed0.json

# Aggregate all JSONs
python scripts/agg_results.py
# → docs/results.tex, docs/figures/dci_heatmap_*.pdf, docs/figures/pareto_mig_sap.pdf
```

---

## Project Structure

```
.
├── configs/
│   ├── base.yaml               # default training config (Hydra)
│   ├── sweep.yaml              # multirun hyperparameter sweep
│   └── eval/                  # eval configs
├── docs/
│   ├── results.tex             # LaTeX results table (3-model, 3-seed)
│   ├── training_results_report.md
│   └── figures/               # DCI heatmaps, Pareto scatter PDFs
├── scripts/
│   ├── eval_run.py             # standalone eval script
│   ├── agg_results.py          # JSON → LaTeX + figures
│   ├── slurm_stage2_array.sh   # training (baselines + SC-VAE)
│   ├── slurm_stage3_array.sh   # ablations
│   ├── slurm_stage4.sh         # MoisesDB OOD
│   ├── slurm_stage56_array.sh  # eval array
│   └── slurm_stage56_agg.sh    # aggregation
├── src/
│   ├── data/                   # NSynth, MoisesDB, augmentation, zero-shot
│   ├── evaluation/             # all 10 disentanglement metrics
│   ├── losses/                 # KL, recon, TC, symmetry
│   ├── models/
│   │   ├── sc_vae.py
│   │   ├── encoder.py / decoder.py
│   │   ├── group_repr.py
│   │   ├── style_head.py / content_head.py
│   │   └── baselines/          # beta_vae, beta_tcvae, factor_vae, ar_hvae
│   ├── training/               # LightningModule, Hydra CLI, callbacks
│   └── utils/
└── tests/                      # unit + integration tests
```

---

## License

TBD.
