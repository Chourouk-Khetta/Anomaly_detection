# Severity-Aware, Explainable Industrial Anomaly Detection

**Unsupervised defect detection on the full MVTec AD benchmark, extended with a multi-factor severity score and reference-based explanations — plus an honest evaluation of whether the severity idea actually works.**

<p align="left">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/dataset-MVTec%20AD-green.svg" alt="MVTec AD">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="License">
</p>

---

## The problem

Standard industrial anomaly detection answers one question: *is this part defective?* On a real production line that is only half the problem. A shift produces hundreds of flagged parts and an inspector can look at a handful, so the operative question is **which ones first?**

An anomaly score is a poor answer, because it measures distance in feature space, not consequence: a bright speck of dust on a cosmetic surface can out-score a faint hairline crack across a sealing face.

This project builds three unsupervised detectors and adds two things on top:

1. **A severity-aware scoring framework** — each flagged region is decomposed into interpretable factors (anomaly intensity, spatial extent relative to the product, structural relevance of its location, shape compactness, local contrast), fused into a severity in `[0, 1]` and a four-level class (Negligible / Minor / Major / Critical). The output is a **ranked inspection queue**.
2. **Reference-based explanations** — beyond heatmaps and Grad-CAM, the system retrieves the *closest normal patch* from the PatchCore memory bank and shows it beside the anomalous one, with the feature distance attributed across backbone layers. The operator gets something checkable: here is what this spot should look like, and here is how it differs.

---

## Key features

- **Full MVTec AD** — all 15 categories, not a curated subset.
- **Three detectors behind one interface**, chosen because they fail differently: **PatchCore** (coreset memory bank), **PaDiM** (per-position Gaussian), **SSIM-AE** (reconstruction residual).
- **Shared frozen backbone** — WideResNet-50-2, ImageNet features from `layer2` + `layer3`, fp16 autocast.
- **Synthetic severity benchmark** — injects defects (blob, scratch, texture, colour shift) with controlled area and contrast to create a continuous ground-truth magnitude, since MVTec has no severity labels.
- **Three-level explainability** — *where* (annotated anomaly map), *why (model)* (Grad-CAM through the k-NN distance), *why (reference)* (nearest normal patch + per-layer attribution).
- **Ranking-based evaluation** — Spearman ρ, Kendall τ, nDCG@k, rank MAE, plus a leave-one-out factor ablation.

---

## Results

### Detection quality (mean over 15 categories, WideResNet-50-2 @ 288px)

| Detector   | Image AUROC | Pixel AUROC | PRO   |
|------------|:-----------:|:-----------:|:-----:|
| **PatchCore** | **0.986** | **0.983** | **0.693** |
| PaDiM      | 0.946       | 0.977       | 0.683 |
| SSIM-AE    | 0.699       | 0.846       | 0.378 |

PatchCore reaches a perfect 1.000 image AUROC on `bottle`, `hazelnut`, `leather` and `tile`, and its worst category is `toothbrush` at 0.928. These numbers are in line with published PatchCore results, which serves as a correctness check on the pipeline.

### Severity ranking — the honest finding

The severity framework is **robust** — under nuisance transforms (small rotations, brightness, noise) the severity score moves by only **±0.021** on a 0–1 scale, so it measures the defect and not the lighting.

But on the current synthetic benchmark, the multi-factor severity score **did not beat a raw anomaly score** at ranking:

- On the location-aware target, PatchCore severity ρ ≈ **0.52** vs raw ρ ≈ **0.69**; severity won only **2/15** categories (PaDiM 4/15, SSIM-AE 12/15).
- A leave-one-out ablation showed **"intensity only" (≈ the conventional score) matched or beat "full severity"** (ρ 0.66 vs 0.56 against the impact target).

This is a genuine mixed result, and the reasons are instructive — see **Limitations** below. The detection pipeline is solid; the *evaluation of the severity idea* is where the open work sits.

---

## Limitations

These are the load-bearing caveats — read them before trusting any severity number.

- **MVTec has no severity ground truth, so everything rests on a synthetic proxy.** MVTec ships defect masks but no severity labels. The entire severity evaluation is therefore scored against a synthetic magnitude *I* defined and generated. If that proxy misrepresents real-world impact, every ranking metric inherits the error — and because the proxy is dominated by area × intensity, a raw anomaly score already tracks it, leaving little headroom for the location- and shape-aware factors to add signal.
- **A single seed and a single evaluation run.** The full sweep is 15 categories × 3 detectors × **one seed**. There are no confidence intervals and no within-category variance, so small differences between methods should be read as noise, not signal. The reported paired test is a rough guide, not population inference.
- **The manual fusion underperforms a learned model.** Factors are combined with a transparent hand-set weighted sum by default. A learned alternative (Ridge) beats it by **+0.062** Spearman on average — meaning the factors carry signal the hand-set weights fail to combine well. The manual weights are kept as default only because they are auditable.

Additional notes: the pipeline is memory-bound rather than compute-bound (each experiment holds a memory bank / per-position covariances on the GPU); SSIM-AE is unstable on several categories and drags the aggregate down.

---

## Future work

- **Replace the synthetic proxy with real severity signal** — a small set of expert-ranked defects, or cost-weighted labels by defect type, to validate severity beyond a self-defined target.
- **Optimise the queue directly** — swap the correlation-scored fusion for a learning-to-rank objective (e.g. nDCG-optimising) and adopt the learned fusion (Ridge / GBM) as default.
- **Redesign the impact target** so that consequence — location, sealing-face proximity, defect type — drives the ranking rather than raw size.
- **Multiple seeds and confidence intervals** for defensible method comparison.
- **Deployment** — expose the ranked queue and explanation panel as an interactive tool suitable for a real line; add modern backbones/detectors (e.g. Reverse Distillation, EfficientAD).

---

## Getting started

The project ships as a single self-contained notebook.

1. Open the notebook and select a GPU runtime (tuned for an NVIDIA T4).
2. Run the install cell. **If it asks you to restart, restart**, then run all cells.
3. MVTec AD (~4.9 GB) is downloaded automatically; the loader tries several mirrors in order.

**Fast mode (default):** `CONFIG.fast_mode = True` runs a ~10-minute smoke test on one category (`screw`) with a ResNet-18 at 160px — enough to check every stage runs. Fast-mode numbers sit well below full-run values and are not meaningful for comparison.

**Full evaluation:** set `CONFIG.fast_mode = False` for all 15 categories, three detectors and the WideResNet-50-2 backbone at 288px. Expect roughly 60–90 minutes on a T4 after download.

### Configuration choices worth knowing

- **No centre crop** — the usual `Resize(256) → CenterCrop(224)` discards ~23% of the frame; MVTec masks cover the whole image, so we resize straight to a square.
- **288px input** — puts the anomaly map on a 36×36 grid (vs 28×28 at 224px), which matters for thin defects and for the region geometry the severity factors read.

---

## How severity is defined

Synthetic defect magnitude is

```
magnitude = sqrt( area_norm · intensity_norm )
```

with **log-normalised area** (industrial impact grows sub-linearly with size) and a **geometric mean** (a defect that is either tiny *or* barely visible cannot come out critical). This definition is fixed *before* the experiments so the severity model cannot be tuned afterwards to flatter the results.

---

## Acknowledgements

- [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) benchmark dataset.
- PatchCore, PaDiM and SSIM-AE, on whose ideas the detectors are based.

## License

Released under the MIT License — see `LICENSE`.
