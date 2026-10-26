# PRISM: opportunistic osteoporosis screening from lumbar radiographs

Code, fold assignments and out-of-fold predictions for the paper:

> Y. Mourdi, H. El Alaoui El Abdallaoui, and H. Allioui, "PRISM: A Multimodal Attention Framework for Opportunistic Osteoporosis Screening from Lumbar Radiographs," submitted to *IEEE Access*, 2026.

PRISM classifies bone status (normal, osteopenia, osteoporosis) and estimates the DXA T-score from routine lumbar radiographs, optionally complemented by CT and five clinical variables. A frozen radiology foundation encoder (RAD-DINO) provides image tokens; a lightweight head (0.92 M trainable parameters) aggregates image regions, views and modalities by gated attention. All experiments use the public [LUMOS](https://doi.org/10.5281/zenodo.18173664) dataset (796 patients) with patient-level five-fold cross-validation.

| Model (out-of-fold, n = 796) | QWK [95% CI] | Osteoporosis AUC [95% CI] |
|---|---|---|
| OST score (logistic regression) | 0.354 [0.288–0.417] | 0.714 [0.676–0.755] |
| Clinical variables (logistic regression) | 0.410 [0.348–0.469] | 0.737 [0.697–0.777] |
| PRISM, ResNet-50 encoder | 0.619 [0.568–0.664] | 0.846 [0.819–0.873] |
| **PRISM, RAD-DINO encoder (final model)** | **0.701 [0.664–0.735]** | **0.887 [0.864–0.911]** |
| PRISM, nested selection (optimism-corrected) | 0.660 [0.620–0.699] | 0.861 [0.835–0.886] |

## Repository content

```
notebooks/PRISM_LUMOS.ipynb    reference pipeline: preprocessing, features, model, all experiments, figures and tables
scripts/prepare_data.py        builds the patient table (image paths + folds) from the LUMOS download
scripts/reproduce_tables.py    recomputes the paper's results from the published predictions (CPU, ~4 min)
prism/                         model, losses, metrics and statistical tests as an importable package
data/folds.csv                 patient-level fold assignment used in every experiment (patient_id, fold)
results/runs.csv               list of the 43 trained configurations (experiment, encoder, region, implants, ...)
results/predictions/           out-of-fold probabilities and T-scores of each configuration
results/inner_validation_qwk.csv  inner-validation QWK per configuration, seed and fold (nested selection)
results/reference_tables/      tables exported by the notebook for the paper
tests/                         smoke tests of the prism package
```

## 1. Check the published results (no GPU, no images)

Only the LUMOS clinical spreadsheet is needed (labels and T-scores are not redistributed here).

```bash
pip install -r requirements.txt
python scripts/reproduce_tables.py --lumos-xlsx /path/to/lumos_clinical_data.xlsx
```

The script recomputes, from `results/predictions/`, the comparison with the baselines, the encoder benchmark, the ablations, the clinical-encoding and CT-encoder comparisons, the nested selection, calibration, decision-curve net benefit and screening thresholds. It writes them to `results/reproduced/` and prints them next to the values of the paper; bootstrap confidence intervals and p-values use the same seeds as the notebook and match exactly.

## 2. Reproduce everything from the images

**Hardware.** The paper was run on an Apple Silicon Mac (PyTorch 2.8, MPS backend). The notebook selects CUDA, MPS or CPU automatically; CPU works but is slow.

**Data.** Download `lumos_clinical_data.xlsx` and `lumos_npy.zip` (20 GB) from [Zenodo](https://doi.org/10.5281/zenodo.18173664) and unzip them into `data/LUMOS/`:

```
data/LUMOS/lumos_clinical_data.xlsx
data/LUMOS/lumos_npy/lumos_npy/lumos_x/lumos_x_001_0.npy ...
data/LUMOS/lumos_npy/lumos_npy/lumos_ct/lumos_ct_001.npy ...
```

**Steps.**

```bash
pip install -r requirements.txt
python scripts/prepare_data.py --lumos-dir data/LUMOS      # -> data/LUMOS/lumos_master_dataset.csv
cd notebooks && jupyter lab PRISM_LUMOS.ipynb               # Run All
```

`prepare_data.py` excludes the 7 patients without usable images and assigns the folds of `data/folds.csv`. `--regenerate-folds` recomputes them as in the original study (StratifiedKFold, 5 folds, `random_state=42`, stratified on class × CT availability) and yields the same assignment.

The notebook downloads the pre-trained encoders (`microsoft/rad-dino` and `facebook/dinov2-base` from Hugging Face, ImageNet CNNs from `timm`) on first use. It caches preprocessed images and features in `data/LUMOS/cache/` and saves every experiment in `data/LUMOS/results_prism/`; an interrupted run resumes where it stopped. Paper figures and tables are exported to `data/LUMOS/results_prism/article/`.

Useful environment variables: `LUMOS_ROOT` (data folder, default `../data/LUMOS`), `LUMOS_SMOKE=1` (end-to-end test with 1 seed and 2 epochs), `LUMOS_CT` (CT encoder, default `raddino`).

Approximate run times on an Apple Silicon Mac: RAD-DINO feature extraction 15–30 min, 3–8 min per frozen configuration (3 seeds × 5 folds), several hours per fine-tuned configuration. The 24-configuration grid takes 2–4 h.

## Correspondence with the paper

| Paper | Notebook section | Runs (`results/runs.csv`, column `experiment`) |
|---|---|---|
| Table II (settings) | 1. Configuration | – |
| Dataset, leakage guard | 2 | – |
| Preprocessing (polarity, views, spine crop, implants, CT) | 3, 3.1, 5.1, 6, 7 | – |
| Encoders and tokens | 5, 8 | – |
| Model and training objective, Eq. (1)–(4) | 9, 10 | – |
| Experiment 1: design grid and fine-tuning | 11, 12.1 | `exp1_design_grid` |
| Nested selection | 12.2 | uses `inner_validation_qwk.csv` |
| Experiment 2: encoder benchmark | 17 | `exp2_encoder_benchmark` (+ ResNet-50 and RAD-DINO from Exp. 1) |
| Experiment 3: component ablation | 13 | `exp3_ablation` |
| Experiment 4: clinical encoding | 18 | `exp4_clinical_encoding` |
| Experiment 5: CT encoder | 19 | `exp5_ct_encoder` |
| Figures and tables | 14, 16 | `results/reference_tables/` |

## Prediction files

Each file of `results/predictions/` has one row per patient (same order as `data/folds.csv`) with columns `patient_id`, `fold`, `p_normal`, `p_osteopenia`, `p_osteoporosis` (averaged over seeds and horizontal-flip test-time augmentation), `t_pred` (predicted T-score), the same quantities with CT masked at inference (`*_noct`, models with a CT branch) and `pred_cutpoints` (class from ordinal cut-points fitted on the inner validation set; the paper uses the most probable class). File names follow `encoder__region__implants__frozen|finetuned__variant__ct-encoder`.

## Reproducibility notes

- Re-training on another GPU or software version gives close but not bit-identical results (non-deterministic GPU kernels). `scripts/reproduce_tables.py` reproduces the published numbers exactly from the saved predictions.
- CT slices are encoded with RAD-DINO in all runs except Experiment 5 (ResNet-50 and DINOv2 CT encoders) and the fine-tuned `raddino / full+spine / flag` run of Experiment 1, which reuses CT features from ResNet-50 because it was computed before the CT encoder was switched to RAD-DINO (notebook flag `REUSE_FT_RESNET_CT`). The `ct_encoder` column of `results/runs.csv` records the CT encoder of each run.
- The linear-probe baseline (image + clinical) and the late-fusion variant F of Experiment 4 are computed in the notebook only; their values are in `results/reference_tables/`.
- Code comments and notebook section titles are in French; the first notebook cell gives an English map of the sections.

## Data license and citation

LUMOS is distributed under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license; commercial use requires permission from the ethics committee of the host institution. This repository does not redistribute images, labels or DXA measurements; `data/folds.csv` and `results/` are derived from LUMOS and are shared under the same license. The code is released under the MIT license (`LICENSE`).

If you use this code, please cite the paper above and the LUMOS dataset:

```bibtex
@inproceedings{shi2025lumos,
  title     = {LUMOS: A Lumbar Multimodal Osteoporosis Screening Dataset with X-ray and CT images},
  author    = {Shi, Keyue and Shen, Qianqian and Ye, Zhaoming and Jiang, Liangjun and Bu, Jiajun and Wang, Haishuai},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  pages     = {13250--13257},
  year      = {2025},
  doi       = {10.1145/3746027.3758282}
}
```

Contact: Youssef Mourdi, Cadi Ayyad University (y.mourdi@uca.ac.ma).
