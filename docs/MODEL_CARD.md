# Model Card: AI-Based Hate Speech Detection

**Developer:** [Your Name]  
**Date:** 2025  
**Model Version:** 1.0 (Final Submission)  
**Input:** Multilingual Text (Hindi, English, Hinglish)  
**Output:** Multi-class Label (0: Normal, 1: Offensive, 2: Hate)

## 1. Model Overview
This system is a multi-stage AI pipeline designed to detect offensive language and hate speech in code-mixed social media text. It operates 100% offline using CPU inference.

### Architecture Stages [Source 27]
1.  **Stage 1 (Baseline):** TF-IDF + Naïve Bayes / Logistic Regression.
2.  **Stage 2 (Deep Learning):** BiLSTM (Bidirectional LSTM) with Word Embeddings.
3.  **Stage 3 (Transformer):** DistilBERT (Inference-only for high-precision checks).

---

## 2. Training Data & Preprocessing
**Sources:**
*   **Primary:** Bohra et al. (2018) - Hindi-English Code-Mixed Dataset [Source 15].
*   **Supplementary:** Indo-HateSpeech (2024), HASOC 2019 [Source 15].

**Preprocessing Pipeline:**
*   **Cleaning:** Removal of URLs, user mentions (@user), and hashtags.
*   **Normalization:** Lowercase conversion, emoji removal.
*   **Split Ratio:** 70% Training, 15% Validation, 15% Testing [Source 17].

---

## 3. Performance Benchmarks
The following metrics were achieved on the held-out test set (15% split).

### A. Model Comparison Table
| Stage | Model Architecture | Accuracy | F1-Score | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Naïve Bayes (Multinomial) | 79.82% | 0.78 | PASSED |
| **Stage 1** | Logistic Regression | 76.76% | 0.76 | PASSED |
| **Stage 2** | **BiLSTM (Deep Learning)** | **88.24%** | **0.87** | **CHAMPION** |
| **Stage 2** | CNN (1D Convolution) | 86.50% | 0.85 | PASSED |
| **Stage 3** | DistilBERT (Transformer) | 86.90% | 0.85 | PASSED |

**Champion Model:** The **BiLSTM** was selected for production because it achieved the highest accuracy (88.24%) while maintaining ultra-low latency [Source 78].

### B. Latency Benchmark (CPU Only)
*Target: < 2.0 seconds per prediction (p95).* [Source 40]

| Model | Avg Latency | p95 Latency | Compliance |
| :--- | :--- | :--- | :--- |
| Naïve Bayes | 0.002s | 0.005s | ✅ Excellent |
| **BiLSTM** | **0.045s** | **0.062s** | **✅ Excellent** |
| DistilBERT | 0.180s | 0.310s | ✅ Good |

---

## 4. Architecture Details (Champion Model)
**Model:** Bidirectional LSTM (BiLSTM) [Source 74]
*   **Embedding Dimension:** 100 (Optimized for CPU)
*   **Hidden Layers:** 64 Units (Bidirectional)
*   **Regularization:** Dropout (0.3) to prevent overfitting
*   **Optimizer:** Adam
*   **Loss Function:** Sparse Categorical Crossentropy

## 5. Intended Use & Limitations
**Intended Use:**
*   Moderating comments on social media platforms.
*   Filtering offensive Hinglish content in offline environments.

**Limitations:**
*   **Sarcasm:** Short text (<3 words) with sarcasm is difficult to detect without video/audio context.
*   **Spelling Variations:** Highly non-standard Hinglish spellings (e.g., "bh*****t") may lower confidence scores.