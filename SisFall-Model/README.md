# Activity-Routed Fall Detection & Anomaly Scoring (SisFall)

A three-stage pipeline that turns raw wearable accelerometer/gyroscope signals into time-frequency images, classifies the activity being performed, and scores each window for fall-like anomalies **relative to that specific activity** — rather than against one global "normal."

Built and evaluated on the [SisFall](http://sistemic.udea.edu.co/en/research/projects/english-falls/) wearable-sensor dataset.

---

## How it works

1. Every 5-second sensor window is converted into **two purpose-built mosaic images** from the same underlying time-frequency transforms (nothing is computed twice).
2. An **EfficientNetV2-S** classifier looks at one mosaic and predicts which daily activity is happening (walking, jogging, sitting, etc.).
3. A **PatchCore** anomaly detector — trained separately per activity group — checks the second mosaic against that activity's own memory bank of "normal" patterns, and produces an anomaly score.
4. The classifier's predicted activity determines *which* PatchCore model and threshold a window is scored against (this routing/fusion step is designed in the project plan; see [Status & Roadmap](#status--roadmap)).

```
Raw sensor window (5s, Acc1 + Gyro, 200Hz)
        │
        ├──► Mosaic_ADL_2x2 (384×384)       ──► EfficientNetV2-S ──► predicted activity
        │                                                                   │
        └──► Mosaic_PatchCore_2x2 (456×456) ──► PatchCore_{activity} ◄──────┘
                                                        │
                                                anomaly score + threshold
                                                        │
                                                fall / no-fall decision
```

---

## Repository contents

This repo's three folders correspond to the three stages of the pipeline, in order:

| # | Folder | Role |
|---|---|---|
| 1 | `Dataset to image generation model/` | Converts raw SisFall sensor recordings into the two mosaic image datasets used by both downstream models. |
| 2 | `EfficientNetV2-S Model/` | Trains and evaluates the activity classifier on the ADL mosaics. |
| 3 | `PatchCore model/` | Builds and evaluates one anomaly-detection model per activity group on the PatchCore mosaics. |

```
SisFall-Model/
  Dataset to image generation model/
    SisFall_RGB_Mosaic_images_Pipeline.py
  EfficientNetV2-S Model/
    sisfall-efficientnetv2-s-train2-advanced2.ipynb
  PatchCore model/
    sisfall-patchcore-train5.ipynb
```

**Run order:** Stage 1 must be run first (it produces the mosaic images both models consume). Stages 2 and 3 are independent of each other and can be run in either order — both read from Stage 1's output, neither depends on the other's output. A script tying the two trained models into one live routing pipeline is not yet part of this repo (see [Status & Roadmap](#status--roadmap)).

---

## The dataset: SisFall

[SisFall](http://sistemic.udea.edu.co/en/research/projects/english-falls/) contains recordings from **38 subjects** (23 young adults, 15 elderly), each performing 19 daily activities (`D01`–`D19`) or 15 simulated fall types (`F01`–`F15`), sampled at **200 Hz** from two accelerometers (ADXL345, MMA8451Q) and one gyroscope (ITG3200).

> **Note on elderly subjects:** per SisFall's own documentation, elderly subjects performed only simulated ADLs, not simulated falls (with one exception). Fall data in this dataset is almost entirely adult-simulated — a real limitation worth stating in any write-up based on this work.

### Activity labels

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

## Stage 1 — `Dataset to image generation model/`

**`SisFall_RGB_Mosaic_images_Pipeline.py`**

Computes three time-frequency representations per sensor axis, per 5-second window, for Acc1 and Gyro only (Acc2 — a second waist accelerometer — is dropped as redundant with Acc1), and recombines them into two mosaics without ever recomputing the underlying transforms twice.

**Representations computed (once, shared between both mosaics):**
- **Scalogram** — Continuous Wavelet Transform (Morlet, scales 1–128), `viridis` colormap. Captures both slow rhythmic motion and sharp impacts.
- **Spectrogram** — Short-Time Fourier Transform (`nperseg=256, noverlap=128`), `inferno` colormap. Captures regular cadence patterns (walking, jogging).
- **Kurtogram** — sliding-window excess kurtosis on the CWT magnitude, computed only over scales 8–128 (the lowest 8 scales are excluded as noise-dominated), `cividis` colormap — chosen because its monotonic luminance preserves the *sign* of kurtosis through grayscale conversion. Specifically designed to highlight sudden, impulsive events (impacts) against otherwise calm motion.

**Windowing:** 5-second windows, 50% overlap, with 3 seconds of extra signal context attached before transform and trimmed after — this removes artificial ringing at window edges without inventing data. Only full-length windows are kept.

**Global colour bounds** are computed from Daily Living (ADL) files only — never from Fall files — so a genuine fall naturally saturates against a scale calibrated purely on normal activity, which is what makes it visually stand out.

**The two mosaics produced:**

| Variant | Layout | Cell size | Canvas | Feeds |
|---|---|---|---|---|
| `Mosaic_ADL_2x2` | rows = [Acc1, Gyro], cols = [scalogram, spectrogram] | 188×188 | 384×384 | EfficientNetV2-S |
| `Mosaic_PatchCore_2x2` | rows = [Acc1, Gyro], cols = [scalogram, kurtogram] | 224×224 | 456×456 | PatchCore |

Every representation is rendered once at a shared 224×224 cache resolution; each mosaic variant does its own cheap resize at final assembly — the CWT/STFT/kurtosis computation and the matplotlib render only ever run once per device per window, regardless of how many mosaic variants are configured.

**Running it:**
```bash
# Full run — generates both mosaic datasets
python SisFall_RGB_Mosaic_images_Pipeline.py "/path/to/SisFall-Model"

# Scan-only — prints fresh colour bounds without writing any images.
# Re-run this whenever the kurtogram scale range, CWT scales, STFT settings,
# or which devices are used changes, since all of those shift the value
# distributions the bounds are calibrated to.
python SisFall_RGB_Mosaic_images_Pipeline.py "/path/to/SisFall-Model" --bounds-only
```

Expected raw data layout: `<project_root>/SisFall_dataset/{SA01..SA23, SE01..SE15}/*.txt`

**Output:**
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

## Stage 2 — `EfficientNetV2-S Model/`

**`sisfall-efficientnetv2-s-train2-advanced2.ipynb`**

Trains a 15-way activity classifier on `Mosaic_ADL_2x2` (the 19 original D-codes are merged into 15 classes — e.g. `Jogging_Slow`/`Jogging_Fast` → `Jogging`; the four sitting variants → `Sitting` — based on which classes an earlier run was observed confusing with each other).

**Training recipe:**
- **EfficientNetV2-S**, ImageNet-pretrained, fine-tuned in stages: head-only → last 3 blocks unfrozen → last 6 blocks unfrozen, each at a progressively lower learning rate, followed by Stochastic Weight Averaging (SWA).
- **AdamW** optimizer, **label smoothing**, **softened class weighting** (α = 0.5) plus a `WeightedRandomSampler` to correct for very uneven class sizes (e.g. `Jogging`: 2,942 images vs. `Gentle_Jump`: 360).
- **Safe augmentation only** — `RandomResizedCrop` + `RandomErasing`. Deliberately **no flip, no rotation, no colour/hue jitter**, since flipping or rotating a motion-signal image would invent a physically impossible movement, and colour jitter would distort the colormap's real signal-magnitude encoding.
- **3-crop test-time augmentation (TTA)** at final evaluation.
- Evaluated with a full classification report, confusion matrix, per-class precision/recall/specificity/F1, Top-k accuracy, and **Grad-CAM** explainability (verifying the model attends to real signal content in the mosaic, not image borders or gutters).

**Dataset used:** 16,437 generated mosaic images across 38 subjects, 15 merged classes.

> **Split methodology — please read before citing these numbers as "generalizes to new people":** the current split (`[INFO] Window-wise split sizes...`) pools all subjects' windows together and splits them individually — every one of the 38 subjects appears in train, validation, **and** test. This is **not** a subject-wise (leave-subjects-out) split, even though an axis label in one saved chart says "Subject-wise Test." The numbers below reflect strong in-distribution performance; a true subject-wise re-validation is listed as follow-up work in [Status & Roadmap](#status--roadmap).

### Results (final run, 3-crop TTA, 2,477 held-out test windows)

| Metric | Value |
|---|---|
| Test accuracy | **96.65%** |
| Macro-F1 | **0.9564** |
| Weighted F1 | **0.9666** |
| Top-2 accuracy | 99.64% |
| Top-3 accuracy | 99.92% |

**Strongest classes (100% recall):** `Walking_Slow`, `Back_Lateral_Change`, `Stumble_Walking`.
**Weakest class:** `Standing_KneeBend` — 83.9% precision, 90.7% recall (still frequently confused with `Sitting`-adjacent activities).

**Output artifacts:**
```
final_efficientnetv2s_adl.pt / _ema.pt / _swa.pt   — trained checkpoints
training_curves_final.png                           — loss / accuracy / error / macro-F1
confusion_matrix_final.png                           — raw counts, test set
confusion_matrix_normalized_final.png                 — row-normalized %, test set
weak_group_confusion_final.png                        — sub-confusion matrix for the harder activity cluster
per_class_accuracy_bar_final.png                      — per-class accuracy, weak group highlighted
gradcam_weak_group_final.png                           — explainability: which mosaic region drove each decision
classification_report_final.txt, per_class_metrics_final.csv
split_train.csv / split_val.csv / split_test.csv       — exact reproducible split
```

---

## Stage 3 — `PatchCore model/`

**`sisfall-patchcore-train5.ipynb`**

Builds **one PatchCore anomaly-detection model per activity group**, each trained *only* on that activity's normal ("good") windows — no fall data is ever used to train a memory bank, which is the core appeal of a one-class approach for a domain where labeled falls are scarce and risky to collect from the people who need protection most (the elderly).

This notebook runs five stages, in order, within itself:

1. **Dataset organizer** — reorganizes `Mosaic_PatchCore_2x2` into four per-activity folders (`train/good`, `test/good`, `test/anomaly`), using SisFall's own official fall-type descriptions to route each of the 15 fall codes to the activity it's conceptually closest to:

   | Activity group | Daily-living sources | Fall codes routed here |
   |---|---|---|
   | Walking | Walking_Slow, Walking_Fast | F01, F02, F03, F04, F06, F07 |
   | Jogging | Jogging_Slow, Jogging_Fast | F05 |
   | Sitting | 4 sit variants (half/low height × slow/fast) | F10, F11, F12, F13, F14, F15 |
   | Standing_GettingUp | Collapse_Into_Chair | F08, F09 |

2. **Anomaly window refiner** — a 15-second fall trial with 50% overlap produces ~5 windows, but only 1–2 actually contain the impact; the rest are genuinely normal pre-/post-fall activity that would otherwise pollute threshold calibration. This stage keeps only the top-2 highest-kurtogram-intensity windows per trial as true "anomaly" labels, routes clearly-early windows to a separate `prefall_context` evaluation set, and excludes the ambiguous windows immediately around the impact.

3. **PatchCore trainer** — no gradient descent, no loss function, no backward pass. A **frozen, ImageNet-pretrained WideResNet-50** (`layer2` + `layer3` hooks) extracts local patch features from every `train/good` image. A **greedy coreset** (farthest-point sampling, 2% retention, capped at 2,000 points) builds the memory bank — built **separately per original sub-activity** (e.g. `Sit_HalfHeight_Slow`, `Sit_HalfHeight_Fast`, ... each get their own mini coreset before being concatenated), which guarantees every sub-activity gets fair representation rather than hoping a single pooled coreset balances itself. At test time, a window's anomaly score is the worst (maximum) nearest-neighbour distance across all its patches.

4. **Threshold calibration — leakage-safe by design.** `test/good` and `test/anomaly` are split 50/50 into a *calibration* half (used only to pick the threshold) and a *final held-out* half (used only to report results). The naive alternative — picking the threshold as the lowest anomaly score seen among fall test images — would mathematically guarantee 100% recall on that same test set, which is not a real result. This script avoids that trap.

5. **Heatmap visualization + manual verification tool** — per-window anomaly heatmaps on a shared, fixed colour scale (so a genuinely calm window doesn't falsely look "hot" just because matplotlib auto-scaled to its own min/max), with the peak location labeled by mosaic quadrant (e.g. `Acc1-Kurtogram`), plus a spot-check tool that lets you visually confirm the window-selection heuristic from stage 2 picked the right window.

### Results (final run, held-out test set, threshold never seen during calibration)

| Activity | Recall | Specificity | FPR | Trial-level recall | ROC-AUC |
|---|---|---|---|---|---|
| **Walking** | 99.86% | 91.97% | 8.03% | 99.81% (538/539) | **0.9996** |
| **Jogging** | 100.00% | 90.14% | 9.86% | 100.00% (89/89) | **0.9992** |
| **Sitting** | 99.86% | 3.56% | 96.44% | 100.00% (547/547) | 0.8072 |
| **Standing_GettingUp** | 100.00% | 14.04% | 85.96% | 100.00% (176/176) | 0.7568 |

**What this means, honestly:**
- **Fall recall is excellent across all four activities** (≥99.86%) — the system essentially never misses a fall in its own held-out test set, including at the trial level.
- **Walking and Jogging generalize very well** — both specificity (>90%) and ROC-AUC (>0.999) are strong, meaning the model reliably tells real falls apart from normal walking/jogging motion.
- **Sitting and Standing_GettingUp still have a real, unresolved specificity problem** — both flag the large majority of genuinely normal activity as a fall (96.4% and 86.0% false-positive rates respectively), even after two targeted fixes (window-label refinement and per-sub-activity balanced memory banks) were applied specifically to address this. The notebook's own analysis traces part of this to how visually similar a genuine sit-down/get-up motion is to a sit-related fall in this mosaic representation — this remains open work, not a solved problem, and should be reported as such.
- **Pre-fall-context specificity is low everywhere** (5.9%–22.0%) — the model is noticeably less confident that the moments right before a fall are "normal" than it is about ordinary daily activity, which is a realistic and informative finding rather than a bug: those windows are, by definition, closer to the edge of normal.

**Output artifacts:**
```
patchcore_models/
  memory_bank_{Activity}.pt           — trained one-class model per activity (Walking, Jogging, Sitting, Standing_GettingUp)
  patchcore_{Activity}_analysis.png   — score histogram + ROC curve
  patchcore_summary.csv               — all 4 activities' final metrics (the table above)
  heatmap_{Activity}.png              — overview anomaly-heatmap grid per activity
manual_verification/
  {Activity}_{FallCode}_{Subject}_{Trial}.png  — spot-check visualizations of the window-selection heuristic
```

---

## ⚠️ Known limitation: good/good split inconsistency

The dataset-organizer stage's own docstring states the intent is a **subject-wise** split for `train/good` vs. `test/good` (the same justification used for the classifier: a test set should never contain a subject the model's memory bank already saw). However, the function actually used (`window_wise_good_split`) implements an **image-level, window-wise** split instead — it pools every subject's images together and splits by individual image, confirmed directly in the notebook's run log: *"train/good subjects (38): [...all 38 subjects...] / test/good subjects (38): [...the same 38 subjects...]"*. The same subject therefore appears in both the memory bank and its own test set.

This is a genuine data-leakage risk for the specificity numbers reported above (particularly plausible as a *partial* explanation for why Walking/Jogging's specificity is high — if some of what's being "recognized" as normal is a specific subject's gait signature rather than a fully general pattern). **Before reporting these specificity/FPR numbers as final, fixing this to a true subject-wise split and re-measuring is strongly recommended.**

---

## Status & Roadmap

**Completed and evaluated:**
- Stage 1 — mosaic generation pipeline (both variants).
- Stage 2 — EfficientNetV2-S activity classifier, fully trained and evaluated (96.65% accuracy).
- Stage 3 — four activity-specific PatchCore anomaly detectors, fully trained and evaluated (table above).

**Not yet implemented:**
- **Live routing / fusion stage** — tying the two trained models together into one script that classifies a new window's activity, routes it to the matching PatchCore model, and applies that activity's calibrated threshold. The logic for this is documented (see the trainer notebook's own "Integration Example" section) but not yet packaged as runnable code in this repo.
- **Severity estimation (mild / moderate / severe)** — not present in any notebook in this repo yet. Planned to be derived from anomaly-score magnitude, heatmap spread, and kurtogram peak intensity, but not yet implemented.
- **Subject-wise re-validation** for both the classifier (Stage 2) and the PatchCore good/good split (Stage 3) — current numbers are window-wise/in-distribution; a true leave-subjects-out re-run is needed before claiming generalization to unseen individuals.
- **Fallback policy for unmonitored activities** — only 4 of the classifier's 15 activity classes currently have a matching PatchCore model (Walking, Jogging, Sitting, Standing_GettingUp). A policy for what happens when the classifier predicts one of the other 11 activities (e.g. Stairs, Car_InOut, Standing_KneeBend) is not yet defined.
- **Sitting / Standing_GettingUp specificity problem** — still unresolved after two targeted fixes; needs further investigation (see Stage 3 results above).

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

Both notebooks were written for a Kaggle GPU environment (CUDA available, `/kaggle/input` / `/kaggle/working` paths with auto-detection fallback) but will run anywhere with the paths adjusted.
