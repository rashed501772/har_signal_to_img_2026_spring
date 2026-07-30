# Activity-Routed Ensemble for Fall Detection & Severity Estimation (SisFall)

Dual time-frequency mosaic pipeline + activity classifier + per-activity anomaly detection ensemble for fall detection and severity estimation on the [SisFall](http://sistemic.udea.edu.co/en/research/projects/english-falls/) wearable-sensor dataset.

Instead of one model trying to spot a fall directly, this project **first classifies what activity a person is doing**, then routes that window to **an anomaly detector trained only on that specific activity's normal patterns** — the idea being that "abnormal" only means something relative to what a person was actually doing at the time.

---

## How it works

1. Every 5-second sensor window is turned into **two different mosaic images** from the same underlying transforms (no data is processed twice).
2. An **EfficientNetV2-S** classifier looks at one mosaic and predicts which of 19 daily activities is happening (walking, sitting down, stumbling, etc.).
3. That prediction routes the window to a **matching PatchCore anomaly detector**, trained only on that one activity's normal windows — it checks the second mosaic for anything that doesn't look like "normal `<activity>`."
4. If flagged abnormal, the anomaly score, heatmap spread, and impact sharpness combine into a mild/moderate/severe severity estimate.

```
Raw sensor window (5s, Acc1 + Gyro, 200Hz)
        │
        ├──► Mosaic_ADL_2x2 (384×384)  ──► EfficientNetV2-S ──► predicted activity
        │                                                              │
        └──► Mosaic_PatchCore_2x2 (456×456) ──► PatchCore_{activity} ◄─┘
                                                        │
                                                anomaly score + heatmap
                                                        │
                                                severity tier (mild/moderate/severe)
```

---

## Repository contents

| File | What it does |
|---|---|
| `SisFall_RGB_Mosaic_images_Pipeline.py` | Reads raw SisFall `.txt` sensor files and generates the two mosaic image datasets described below. |

| `train_efficientnetv2s_adl.py` | Trains the EfficientNetV2-S activity classifier on `Mosaic_ADL_2x2`, with the full evaluation suite (confusion matrix, per-class precision/recall/specificity/F1). Written for Kaggle (GPU notebook), but runs anywhere with the right paths. |

**Not yet in this repo:** the PatchCore per-activity training script (Stage 5 of the plan). The classifier stage is complete and evaluated; the anomaly-detection stage is designed and documented in the plan but not yet implemented in code here.

---

## The dataset: SisFall

[SisFall](http://sistemic.udea.edu.co/en/research/projects/english-falls/) contains 4,510 recordings from 38 subjects (23 adults, 15 elderly), each performing one of 19 daily activities (`D01`–`D19`) or one of 15 simulated fall types (`F01`–`F15`), recorded at 200 Hz from two accelerometers (ADXL345, MMA8451Q) and one gyroscope (ITG3200).

> **Note on elderly subjects:** per SisFall's own documentation, elderly subjects performed only simulated ADLs, not simulated falls (with one exception — a Judo-trained elderly subject who also performed falls). Fall data in this dataset is almost entirely adult-simulated. This is a real limitation worth stating explicitly in any write-up based on this work.

### Activity labels (corrected to match the official SisFall documentation)

| Code | Activity | Code | Activity |
|---|---|---|---|
| D01 | Walking slowly | D11 | Collapse into chair |
| D02 | Walking quickly | D12 | Lying down slowly, sit again |
| D03 | Jogging slowly | D13 | Lying down quickly, sit again |
| D04 | Jogging quickly | D14 | Back ↔ lateral position change |
| D05 | Stairs, slowly | D15 | Standing, knee bend |
| D06 | Stairs, quickly | D16 | Standing, no knee bend |
| D07 | Sit, half-height chair, slowly | D17 | Get in/out of car |
| D08 | Sit, half-height chair, quickly | D18 | Stumble while walking |
| D09 | Sit, low chair, slowly | D19 | Gentle jump |
| D10 | Sit, low chair, quickly | | |

---

## Pipeline stage 1 — mosaic generation

`SisFall_RGB_Mosaic_images_Pipeline.py` computes three time-frequency representations per sensor axis, per 5-second window, per device (Acc1 and Gyro only — see below), and recombines them into two purpose-built mosaics without ever recomputing the underlying transforms twice.

**Representations computed (once, shared between both mosaics):**
- **Scalogram** — Continuous Wavelet Transform (Morlet, scales 1–128), `viridis` colormap.
- **Spectrogram** — Short-Time Fourier Transform (`nperseg=256, noverlap=128`), dB-converted, `inferno` colormap.
- **Kurtogram** — sliding-window excess kurtosis on the CWT magnitude, computed only over scales 8–128 (the lowest 8 scales are excluded as noise-dominated), `cividis` colormap chosen specifically because its monotonic luminance preserves the *sign* of kurtosis (impulsive vs. very-regular) through grayscale conversion — a diverging colormap like `coolwarm` would collapse that distinction.

**Why Acc1 + Gyro only, not Acc2:** Acc2 is a second waist-mounted accelerometer, largely redundant with Acc1 at the same mount location. Dropping it leaves Gyro (angular velocity) as the genuinely complementary signal to Acc1 (linear acceleration), and makes both mosaics square — so resizing never distorts the aspect ratio.

**The two mosaics:**

| Variant | Layout | Cell size | Canvas | Feeds |
|---|---|---|---|---|
| `Mosaic_ADL_2x2` | rows = [Acc1, Gyro], cols = [scalogram, spectrogram] | 188×188 | 384×384 | EfficientNetV2-S (native input size, no resize needed) |
| `Mosaic_PatchCore_2x2` | rows = [Acc1, Gyro], cols = [scalogram, kurtogram] | 224×224 | 456×456 | PatchCore (finer patch grid for anomaly localization) |

Every representation is rendered once at a shared 224×224 cache resolution; each mosaic variant then does its own cheap resize (188×188 for ADL, unchanged for PatchCore) at the final assembly step — the CWT/STFT/kurtosis math and the matplotlib render only ever run once per device per window, regardless of how many mosaic variants are configured.

**Global colour bounds** are computed from Daily Living (ADL) files only — Fall files are never used to set the color scale, so a genuine fall naturally saturates against a scale calibrated purely on normal activity, which is what makes it visually stand out as anomalous.

### Running the generator

```bash
# Full run — generates both mosaic datasets
python SisFall_RGB_Mosaic_images_Pipeline.py "/path/to/SisFall-Model"

# Scan-only — prints fresh vmin/vmax bounds without writing any images.
# Run this whenever KURT_MIN_SCALE_IDX, CWT_SCALES, STFT settings, or
# MOSAIC_ROW_DEVICES changes, since all of those shift the underlying
# value distributions the bounds are calibrated to.
python SisFall_RGB_Mosaic_images_Pipeline.py "/path/to/SisFall-Model" --bounds-only
```

Expected raw data layout: `<project_root>/SisFall_dataset/{SA01..SA23, SE01..SE15}/*.txt`

Output:
```
Generated_Images_Mosaic/
  Mosaic_ADL_2x2/
    Daily Living/{activity}/{subject}/{stem}.png
    Fall/{subject}/{stem}.png
  Mosaic_PatchCore_2x2/
    Daily Living/{activity}/{subject}/{stem}.png
    Fall/{subject}/{stem}.png
```

---

## Pipeline stage 2 — ADL activity classifier (EfficientNetV2-S)

`train_efficientnetv2s_adl.py` trains a 19-way classifier on `Mosaic_ADL_2x2`, at its native 384×384 resolution — chosen specifically because it matches EfficientNetV2-S's own pretraining resolution, so no dataloader-side resize is needed.

**What the script does, in order:**

1. Indexes every image under `Mosaic_ADL_2x2/Daily Living/{activity}/{subject}/*.png` into a manifest (filepath, class, subject).
2. Splits **class-wise 70/15/15** (each activity independently divided into train/val/test) with a fixed seed, and saves the exact split as CSVs (`split_train.csv`, `split_val.csv`, `split_test.csv`) for auditability.
3. Applies **class-weighted cross-entropy loss** to correct for activity classes having different amounts of recorded data (single 100s trials for D01–D04 vs. 5×12s trials for most others).
4. Loads `EfficientNetV2-S` with ImageNet weights, replaces the classifier head with a 19-way linear layer.
5. **Two-stage fine-tuning**: Stage 7a trains only the new classifier head (backbone frozen, 8 epochs, `lr=1e-3`); Stage 7b unfreezes the last 3 backbone blocks and continues at a lower rate (`lr=1e-5` backbone / `1e-4` head, up to 25 epochs, cosine LR schedule, early stopping on validation accuracy with patience 7).
6. Deliberately **no horizontal flip and no color/hue jitter** — flipping would reverse the time axis (breaking causal before/after structure), and color jitter would distort the colormap's real signal-magnitude encoding.
7. Evaluates on the held-out test split: full classification report, confusion matrix heatmap, and a per-class table of TP/TN/FP/FN/precision/recall(sensitivity)/specificity/F1 — with **D18 (stumble) and the chair-height/lying-down activity pairs flagged explicitly**, since these are the classes most likely to matter for downstream PatchCore routing accuracy.

### Running it

Written for a Kaggle GPU notebook (T4×2 or P100, internet on). Update `DATASET_SLUG` / `DATA_ROOT` near the top to match your uploaded dataset's path — if the guess is wrong, it auto-searches `/kaggle/input/*/Generated_Images_Mosaic/Mosaic_ADL_2x2/Daily Living` and uses whatever it finds.

To run outside Kaggle, just change `DATA_ROOT` to point at your local `Generated_Images_Mosaic/Mosaic_ADL_2x2` folder and `OUTPUT_DIR` to wherever you want checkpoints/artifacts saved.

**Output artifacts:**
```
best_efficientnetv2s_adl.pt     — best checkpoint by validation accuracy
training_curves.png             — loss / accuracy / error curves
confusion_matrix.png            — 19x19 confusion matrix heatmap
classification_report.txt       — sklearn classification_report
per_class_metrics.csv           — TP/TN/FP/FN/precision/recall/specificity/F1 per activity
split_train.csv / split_val.csv / split_test.csv  — exact reproducible split
```

---

## Roadmap / not yet implemented

- **PatchCore per-activity anomaly detection (Stage 5 of the plan)** — one anomaly model per activity (up to 19), trained only on that activity's `Mosaic_PatchCore_2x2` "normal" windows, using a frozen WideResNet-50 backbone and coreset-subsampled memory bank. Recommended training order (see plan for full rationale): Group 1 (D01–D04) → Group 2 (D05, D06, D15, D16) → Group 3 (D07–D10, D17) → Group 4 (D11–D14, D18, D19 — near-fall activities, trained last and validated separately for specificity against real falls).
- **Classifier-routed inference** (Stage 6) — tying the two trained models together: classify activity, route to the matching PatchCore model, combine anomaly score + heatmap + kurtogram peak into a severity tier.
- **End-to-end validation** (Stage 7) — routing correctness, fall detection ROC-AUC/F1, near-fall specificity check, severity ranking sanity check.

See `Complete_Pipeline_Plan_v5_corrected.md` for the full design rationale, open decisions (e.g., whether Group 4 gets full independent PatchCore models or is used purely as a specificity test set), and validation strategy for each of these.

---

## Requirements

```
numpy
pandas
scipy
matplotlib
seaborn
PyWavelets
Pillow
scikit-learn
torch
torchvision
```


