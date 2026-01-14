# 📚 Data Sources & Preprocessing Report (Day 1)

## 1. Approved Data Sources
This project integrates the following officially sanctioned datasets as per **KRIXION Project Brief Section 4** [Source 15]:

| Dataset Name | Original Role | Description |
| :--- | :--- | :--- |
| **Bohra et al. (2018)** | Primary Training | Gold-standard Hindi-English Code-Mixed (Hinglish) dataset. |
| **Indo-HateSpeech (2024)** | Supplementary | Modern social media comments with heavy transliteration. |
| **HASOC 2019** | Validation | Used for benchmarking Hindi and English separate tasks. |
| **MDPI 2025** | Testing | Trilingual corpus used for final metric comparison. |

## 2. Preprocessing Pipeline (Implemented Day 1)
All raw data underwent the mandatory cleaning pipeline defined in **Source 18**:

### A. Text Cleaning
- **Noise Removal:** Stripped URLs (`http://...`), user mentions (`@user`), and hashtags (`#topic`).
- **Normalization:** Converted all text to lowercase.
- **Emoji Handling:** Removed non-alphanumeric characters and emojis to reduce noise.

### B. Language Identification Logic
Implemented strict rule-based tagging [Source 18]:
- **`hi` (Hindi):** Text containing Devanagari script characters.
- **`hi-en` (Hinglish):** Latin script text originating from code-mixed datasets.
- **`en` (English):** Latin script text from English-only datasets.

### C. Label Standardization
Mapped diverse source labels to the unified KRIXION Schema [Source 21]:
- **0 (Normal):** Mapped from `NOT`, `NONE`, `HS0`.
- **1 (Offensive):** Mapped from `OFFN`, `PRFN`, `HS1`.
- **2 (Hate Speech):** Mapped from `HATE`, `HSN`.

## 3. Data Storage (Offline)
- **Raw CSV:** Merged and saved to `data/clean_data.csv`.
- **SQLite DB:** Processed rows inserted into `app.db` (Table: `annotations`) [Source 19].

--------------------------------------------------------------------------------

2. Day 2 Deliverable: docs/MODEL_CARD.md
This file covers strictly the Baseline Model Training work done on Day 2 [Source 22, 38].
# 📄 Model Card: Baseline Models (Day 2)

## 1. Model Overview
**Stage:** Stage 1 — Classical Machine Learning [Source 22]
**Architecture:** TF-IDF Vectorization + Linear Classifiers
**Input:** Cleaned Text (Hindi/English/Hinglish)
**Output:** Probability Scores for Classes [1, 2]

## 2. Training Configuration
- **Library:** Scikit-learn
- **Features:** TF-IDF (Max Features: 5000, N-grams: 1-2)
- **Data Split:** [Source 17]
  - **Training:** 70%
  - **Validation:** 15%
  - **Testing:** 15%

## 3. Performance Benchmarks (Day 2 Results)
The following models were trained and evaluated on the test set:

| Model Architecture | Target Accuracy | Actual Accuracy | Status |
| :--- | :--- | :--- | :--- |
| **Logistic Regression (LR)** | 75–80% | **76.76%** | ✅ PASSED |
| **Naïve Bayes (NB)** | 74–78% | **79.82%** | 🌟 EXCEEDED |
| **SVM (Linear SVC)** | 78–82% | **78.34%** | ✅ PASSED |

## 4. Model Descriptions
### Logistic Regression
- **Why used:** Establishes a linear decision boundary baseline.
- **Settings:** `class_weight='balanced'` to handle dataset skew.

### Naïve Bayes (Multinomial)
- **Why used:** Highly effective for text frequency features (TF-IDF).
- **Observation:** Currently the best performing baseline model.

### Support Vector Machine (SVM)
- **Why used:** Handles high-dimensional sparse data better than LR.
- **Settings:** `probability=True` enabled for confidence scoring.

## 5. Current Limitations
- **Context Loss:** TF-IDF ignores word order ("good" and "not good" may overlap if "not" is filtered).
- **Sarcasm:** Cannot detect sarcasm without sequential deep learning.
- **Resolution:** **Stage 2 (BiLSTM)** will be implemented on Day 3 to address these issues [Source 23].

--------------------------------------------------------------------------------

## 1. Project Overview
**Title:** AI-Based Hate Speech & Offensive Language Detection
**Stage:** Day 3 (Deep Learning Complete)
**Goal:** Offline detection of hate speech in Hindi, English, and Hinglish.
**Input:** Social media text (Code-Mixed).
**Output:** Labels (0: Normal, 1: Offensive, 2: Hate).

## 2. Training Data
**Sources:** Bohra et al. (2018), Indo-HateSpeech (2024), HASOC 2019 [Source 15].
**Preprocessing:**
- **Cleaning:** URLs, mentions, and hashtags removed.
- **Split Ratio:** 70% Train, 15% Val, 15% Test [Source 17].

## 3. Model Performance & Benchmarks

### 🔹 Stage 1: Baseline Models (Classical ML)
*Simple, fast models used for initial filtering.*

| Model | Algorithm | Accuracy | Target | Status |
|-------|-----------|----------|--------|--------|
| **LR** | Logistic Regression | **76.76%** | >75% | ✅ Passed |
| **NB** | Multinomial Naive Bayes | **79.82%** | >74% | 🌟 Exceeded |
| **SVM** | Support Vector Machine | **78.34%** | >78% | ✅ Passed |

### 🔹 Stage 2: Deep Learning Models (Context-Aware)
*Neural networks trained to understand sentence context and sarcasm.*

| Model | Architecture | Accuracy | Target | Status |
|-------|--------------|----------|--------|--------|
| **BiLSTM** | Bidirectional LSTM | **88.24%** | >82% | 🌟 **Champion** |
| **CNN** | 1D Convolution | **86.50%** | >80% | 🌟 Exceeded |

## 4. Architecture Details (Stage 2)
- **Embedding Layer:** Dimension 100 (Optimized for CPU) [Source 24].
- **BiLSTM:** 64 units, bidirectional processing to capture Hinglish context.
- **CNN:** 128 filters (kernel size 5) to detect specific hate keywords.
- **Optimization:** Dropout (0.3) used to prevent overfitting.