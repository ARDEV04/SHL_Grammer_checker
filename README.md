
# 🧠 **README — SHL Grammar Scoring Engine**

## 📘 **Overview**

This notebook was developed for the **SHL Intern Hiring Assessment 2025**, aiming to build a **Grammar Scoring Engine** for spoken English audio samples.
Each sample is a 45–60 second `.wav` file rated between **0 to 5** based on grammatical quality.

The task:

> Given an audio clip, predict a continuous grammar score that aligns with human MOS (Mean Opinion Score) ratings.

---

## 🎯 **Objective**

To design a **robust multimodal model** that evaluates spoken grammar quality using both **audio and text** signals.

---

## 📂 **Dataset Description**

* **Training Samples:** 409
* **Test Samples:** 197
* **Audio Format:** `.wav` files (each 45–60 seconds)
* **Labels:** Continuous Grammar MOS scores (0–5)
* **Files:**

  * `csvs/train.csv` – filenames + labels
  * `csvs/test.csv` – filenames (no labels)
  * `audios/train/` – training audio files
  * `audios/test/` – test audio files

---

## ⚙️ **Step-by-Step Approach**

### **1️⃣ Initial Baseline: Audio-only (WavLM)**

* Used **Microsoft WavLM-base** model to extract self-supervised (SSL) embeddings.
* Averaged time representations to obtain fixed-size 768-D features per audio.
* Trained a **LightGBM regressor** on these embeddings.

**Results:**

* RMSE ≈ 0.66
* Pearson ≈ 0.50
  → The model captured fluency and clarity but not grammar well.

---

### **2️⃣ Added Whisper ASR + Text Features**

* Used **OpenAI Whisper-small** for transcription.
* Extracted basic **textual statistics** (token count, sentence length, disfluency count, etc.).
* Built a separate **text regression model**.

**Results:**

* RMSE ≈ 0.71
* Pearson ≈ 0.37
  → ASR-transcribed text was clean, so grammar errors were lost; limited correlation.

---

### **3️⃣ Combined Audio + Text (Blended Model)**

* Created a weighted blend of the WavLM and Text models.
* Tuned blend weights to minimize RMSE on out-of-fold predictions.

**Results:**

* RMSE ≈ 0.65
* Pearson ≈ 0.55
  → Slight improvement, indicating complementary signals.

---

### **4️⃣ Major Upgrade — Multimodal Stack Ensemble**

To improve robustness, multiple feature streams were added:

* **WavLM embeddings** (acoustic)
* **HuBERT embeddings** (semantic audio representation)
* **MFCC + Prosody features** (speech rhythm, pauses, energy)
* **Whisper + DeBERTa Text Features**

Each feature set trained its own **LightGBM** base model.
Their outputs were stacked using **Non-Negative Least Squares (NNLS)**, followed by **Isotonic Calibration** for monotonic score mapping.

---

## 🧩 **Model Architecture**

```
           ┌────────────┐
           │   Audio (.wav)   │
           └──────┬─────┘
                  │
     ┌────────────┼────────────┐
     │                         │
 [WavLM-base]            [HuBERT-base]
     │                         │
     └────────────┬────────────┘
                  │
       [MFCC + Prosody Features]
                  │
           ┌──────┴──────┐
           │ LightGBM Base│
           └──────┬──────┘
                  │
         [Stacking Ensemble]
                  │
     [Isotonic Regression Calibrator]
                  │
             Final Grammar Score
```

---

## 📈 **Training Results**

| Model                            | RMSE      | Pearson     |
| -------------------------------- | --------- | ----------- |
| WavLM                            | 0.676     | 0.478       |
| HuBERT                           | 0.626     | 0.581       |
| MFCC/Prosody                     | 0.689     | 0.438       |
| Text (Whisper + DeBERTa)         | 0.675     | 0.475       |
| Concat (All Features)            | 0.629     | 0.574       |
| **Stack Ensemble**               | **0.607** | **0.645**   |
| **Stack + Isotonic Calibration** | **0.549** | **0.697** ✅ |

---

## 🧾 **Key Observations**

✅ WavLM and HuBERT are complementary — combining them improved correlation.
✅ Prosody and MFCC features captured rhythm and pauses that correlate with grammar fluency.
✅ Text model helped refine predictions where ASR captured structural cues.
✅ Stacking and isotonic calibration drastically reduced error variance.

---

## 🧮 **Final Performance**

* **Out-of-Fold (OOF) RMSE:** **0.5491**
* **Out-of-Fold Pearson Correlation:** **0.6972**
---

## 🧰 **Tools and Libraries**

* **Transformers:** Hugging Face (WavLM, HuBERT, DeBERTa, Whisper)
* **Torchaudio:** Audio preprocessing
* **LightGBM:** Gradient boosting for regression
* **Scikit-learn:** CV, metrics, isotonic calibration
* **NumPy / Pandas / Matplotlib:** Data handling and visualization

---

## 🚀 **Improvements Over Time**

| Stage                  | Improvement         | RMSE     | Pearson    |
| ---------------------- | ------------------- | -------- | ---------- |
| Baseline (WavLM only)  | Initial model       | 0.66     | 0.50       |
| + Whisper ASR + Text   | Added text channel  | 0.65     | 0.55       |
| + HuBERT + MFCC        | Multimodal stacking | 0.61     | 0.64       |
| + Isotonic Calibration | Final refined model | **0.55** | **0.70** ✅ |

---

## 🏁 **Final Outcome**

The final system:

* Learns both **acoustic** and **linguistic** cues of grammar.
* Produces **continuous, calibrated grammar scores** between 0 and 5.
* Achieves **excellent generalization** with high rank correlation to human labels.
* Fully automated — takes `.wav` audio as input → outputs a grammar score.

---
