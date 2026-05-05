# Hybrid Quantum–Classical Multimodal Pipeline for PD Classification

This repository contains the implementation for the master's thesis *"Evaluating a Hybrid Quantum–Classical Multimodal Pipeline for Cognitive Performance Classification"* (Tallinn University of Technology, 2026).

The pipeline classifies Parkinson's disease (PD) vs. healthy controls (HC) from digitised spiral drawing data using a combination of tabular kinematic features and image features, evaluated across classical ML models and quantum classifiers (VQC and QCNN).

---

## Overview

Each spiral drawing is represented two ways:

- **Tabular features** — kinematics, pressure, angular motion, tremor (FFT), and pen orientation derived from the raw time-series signal
- **Image features** — 512-dim ResNet18 embeddings of rendered spiral images, reduced to 4 components via PCA

These are fused via early concatenation and fed into classical models (LR, SVM, RF, XGB, DT, KNN) and quantum models (VQC, QCNN), all evaluated under identical nested stratified cross-validation.

---

## Dataset

The primary dataset is **DraWritePD** (54 samples: 20 PD, 34 HC), available from the authors of:

> Valla et al., *"Tremor-related feature engineering for machine learning based Parkinson's disease diagnostics"*, Biomedical Signal Processing and Control, 2022.

The pipeline expects raw JSON trajectory files with six channels per point: `x`, `y`, `t`, `p`, `a` (azimuth), `l` (altitude). CSV files are also supported, with or without the `a` column.

Expected directory structure:
```
PD/
  spiral/
    sample_01.json
    ...
KT/
  spiral/
    sample_01.json
    ...
```

---

## Installation

Python 3.13 recommended. Install dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `scikit-learn`
- `numpy`, `scipy`
- `torch`, `torchvision`
- `qiskit`, `qiskit-machine-learning`, `qiskit-algorithms`
- `opencv-python` (`cv2`)
- `joblib`, `Pillow`, `pandas`, `xgboost`

All quantum experiments run on classical simulation via Qiskit's statevector `Estimator`. No quantum hardware is required.

---

## Pipeline

The full workflow is in `workflow_05052026.ipynb`. The pipeline runs in the following order:

### 1. Load and preprocess data

```python
data, labels = data_prep("./PD/spiral", "./KT/spiral", filetype="json", a_exists=True)
```

- `filetype`: `"json"` or `"csv"`
- `a_exists`: set to `False` if your data does not have the azimuth (`a`) channel

### 2. Extract tabular features

```python
tab_data, feature_labels = data_to_tab(data)
```

Produces ~50–60 features per sample covering velocity, acceleration, jerk, angular motion, pressure dynamics, tremor (FFT 3–12 Hz), and optionally altitude.

### 3. Select tabular features

```python
from sklearn.utils import shuffle
data_shuffled, tab_shuffled, labels_shuffled = shuffle(data, tab_data, labels, random_state=42)

tab_features, selected_names, fisher_scores = feature_ext_tab(
    tab_shuffled, labels_shuffled, feature_labels, k_select=4
)
```

Two-stage selection: variance threshold → Fisher score (`SelectKBest`). Fitted selectors are saved to `tab_feature_selectors.pkl` for reuse at eval time.

### 4. Render spiral images

```python
img_fnames, failed = data_to_img(data_shuffled, labels_shuffled, folder="IMG_drawritepd")
```

Renders 256×256 grayscale PNG images from (x, y) coordinates. Saves to the specified folder.

### 5. Extract and reduce image features

```python
img_raw     = ext_feat_img(img_fnames)          # (n, 512) ResNet18 embeddings
img_features = img_downsize(img_raw, n_components=4)  # (n, 4) via PCA
```

Fitted transform and PCA are saved to `image_transform.pkl` and `img_feature_selectors.pkl`.

### 6. Fuse modalities

```python
features = features_merge(tab_features, img_features)  # (n, 8)
```

### 7. Classical model evaluation (nested CV)

```python
results = run_nested_cv(
    features, labels_shuffled,
    feature_names=feature_labels,
    save_dir="models",
    outer_folds=5
)
```

Runs LR, SVM, RF, XGB, DT, KNN with inner grid search. Prints a summary table, hard-case analysis, and saves per-model confusion matrix PNGs and `nested_cv_results.json`.

### 8. Quantum model evaluation

```python
# VQC
results_vqc = exec_vqc_cv(features, labels_shuffled, epochs=150, k_features=4)

# QCNN
results_qcnn = exec_qcnn_cv(features, labels_shuffled, epochs=150, k_features=4)

# Single train/test run with saved model
model_path = exec_qcnn(features, labels_shuffled, epochs=150)
```

Both quantum models use `k_features=4` (fixing qubits to 4). Labels are encoded as −1 (PD) and +1 (HC). Features are scaled to [0, π] per fold before angle encoding.

---

## Evaluation on a New Dataset (Train on Synthetic, Evaluate on Real)

To evaluate a saved quantum model on a separate dataset:

```python
eval_features, eval_labels = evaluate_model_get_feat(
    "./new_data/PD/", "./new_data/KT/", img_folder="IMG_eval"
)
evaluate_model_predictions("qcnn_model.pkl", eval_features, eval_labels)
```

This applies the saved selectors (`eval=True`) rather than refitting, preventing data leakage.

---

## Saved Artefacts

| File | Contents |
|---|---|
| `tab_feature_selectors.pkl` | Fitted variance threshold + Fisher SelectKBest |
| `image_transform.pkl` | ResNet18 image transform |
| `img_feature_selectors.pkl` | Fitted StandardScaler + PCA |
| `models/<NAME>_final.joblib` | Final classical model fitted on full dataset |
| `qcnn_model.pkl` | Saved QCNN classifier bundle (classifier + x_min/x_max) |

---

## Key Results (DraWritePD, multimodal features)

| Model | Accuracy | F1 |
|---|---|---|
| LR | 0.907 | 0.877 |
| KNN | 0.907 | 0.877 |
| RF | 0.853 | 0.836 |
| QCNN | 0.871 | 0.833 |
| VQC | 0.853 | 0.818 |
| SVM | 0.873 | 0.826 |

Tabular features alone carry most of the discriminative signal. The QCNN is the only model that improves meaningfully when image features are added (tabular F1 0.766 → multimodal 0.833). See the thesis for full ablation results.

---

## Non-Obvious Pitfalls

**Feature selection must stay inside the CV loop.** `feature_ext_tab` fits on training folds only. Fitting on the full dataset before CV inflates performance substantially (see Valla et al.: 92% vs 84% accuracy with vs without proper nesting).

**Derivative chain must be truncated.** Time deltas are floored at 10⁻⁹ s. Features go up to jerk (3rd derivative) for kinematics and acceleration (2nd derivative) for pressure. Beyond this, values reach scales of 10¹⁰–10³² and are not meaningful features.

**Altitude and azimuth behave differently from other channels.** Azimuth varies continuously within a stroke. Altitude is constant within a stroke and updates between strokes. In the guided spiral task, neither survived feature selection — but both are computed and included in the feature pool for generalisability.

**Quantum models output discrete labels.** `predict` returns `{−1, 1}`, so the ROC curve is a single point. `evaluate_model_predictions` uses `predict_proba` if available and falls back gracefully.

**`k_select` in `feature_ext_tab` must not exceed `n_features`.** Pass `k_select=4` explicitly when working with 4–8 features to avoid `SelectKBest` warnings.

---

## Reproducibility

Fixed random seeds (`random_state=42`) are used throughout. Package versions are in `requirements.txt`. All models and evaluation outputs are saved on each run.

For questions about the DraWritePD dataset, contact the original authors (Valla et al., 2022). The pipeline also accepts any JSON or CSV spiral drawing data with the six-channel format described above.

---

## Citation

If you use this pipeline, please cite:

```
Eleriin Rein. "Evaluating a Hybrid Quantum–Classical Multimodal Pipeline for
Cognitive Performance Classification." Master's Thesis, Tallinn University of
Technology, 2026.
```
