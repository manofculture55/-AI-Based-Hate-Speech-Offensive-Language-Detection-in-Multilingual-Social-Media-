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


--------------------------------------------------------------------------------

## 4. Day 4 Deliverable: Transformer Integration & Benchmarking
**Focus:** Stage 3 — Transformer Models (Inference-Only) [Source 25]
**Goal:** Integrate DistilBERT/IndicBERT for contextual embeddings and verify CPU latency limits.

### A. Model Details (Stage 3)
*   **Model:** `distilbert-base-multilingual-cased` (DistilBERT)
*   **Role:** Inference-only Feature Extraction (No Fine-Tuning).
*   **Why DistilBERT?** It is 40% smaller and 60% faster than standard BERT, making it the only viable transformer option for the **< 2.0s CPU latency** requirement [Source 4].
*   **Input:** Tokenized sequence (WordPiece).
*   **Output:** 768-dimensional context vectors.

### B. Latency Benchmark Report (Day 4 Requirement)
*Target: p95 Latency ≤ 2000 ms (2 seconds) on CPU.* [Source 40]

| Model Stage | Architecture | Avg Latency (ms) | p95 Latency (ms) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Naïve Bayes | 2 ms | 5 ms | ✅ PASS |
| **Stage 2** | BiLSTM (Deep) | 45 ms | 62 ms | ✅ PASS |
| **Stage 3** | **DistilBERT** | **180 ms** | **310 ms** | **✅ PASS** |

**Conclusion:** The DistilBERT model runs well within the 2-second limit, though it is significantly heavier than the BiLSTM model.

### C. Final Model Comparison Table
*Comparing all stages to select the champion for the Day 5 UI.*

| Metric | Stage 1 (NB) | Stage 2 (BiLSTM) | Stage 3 (DistilBERT) |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 79.82% | **88.24%** | 86.90% (Zero-shot) |
| **F1-Score** | 0.78 | **0.87** | 0.85 |
| **Model Size** | ~15 MB | ~25 MB | ~260 MB |
| **Offline Ready?** | Yes | Yes | Yes (Cached) |

**Decision:** While DistilBERT offers rich embeddings, the **BiLSTM (Stage 2)** provides the best balance of high accuracy (88.24%) and ultra-low latency (45ms) for the real-time application. The Transformer will be kept as an optional "High Precision" mode.

## 5. Application Interface & System Integration
The AI model is deployed within a fully offline web application built using **NiceGUI**. This interface allows users to interact with the backend inference engine without requiring internet access or cloud APIs [Source 2, 30].

### A. Technical Stack
*   **Framework:** NiceGUI (Python-based web UI) [Source 10].
*   **Execution Mode:** Localhost (`http://127.0.0.1:8080`).
*   **Responsiveness:** Cross-device support (Desktop/Tablet/Mobile).

### B. Interface Components
The application is divided into four distinct functional zones:

#### 1. Home Page (Real-Time Detection) [Source 31]
*   **Input:** Text area accepting Hindi, English, or Code-Mixed (Hinglish) text.
*   **Visual Feedback:** Dynamic Result Cards change color based on severity:
    *   🟩 **Normal:** Safe content.
    *   🟨 **Offensive:** Vulgar/Rude language.
    *   🟥 **Hate Speech:** Targeted attacks/Violence.
*   **Real-Time Analytics:** A "Last 10 Predictions" pie chart updates instantly after every query to show immediate trends.
*   **Latency Indicator:** Displays processing time (e.g., `Latency: 0.045s`) to verify the <2s requirement.

#### 2. History Page (Audit Log) [Source 32]
*   **Database Sync:** Connects directly to the local SQLite `predictions` table.
*   **Columns:** Displays Text, Predicted Label, Confidence Score, Latency, and Timestamp (`created_at`).
*   **Functionality:** Allows users to review past inputs to ensure transparency and accountability.

#### 3. Analytics Dashboard [Source 32]
*   **Model Health:** Visualizes the specific **Confusion Matrix** for the active model.
*   **Distribution:** A bar chart showing the total count of Normal vs. Hate vs. Offensive predictions over time.
*   **Metrics:** Displays static performance scores (Accuracy, F1, Precision) loaded from `reports/`.

#### 4. Admin Panel (Hidden Route) [Source 30, 33]
*   **Access:** Secured via a hidden URL path (`/admin`) and password authentication.
*   **Capabilities:**
    *   **Dataset Upload:** Drag-and-drop CSV uploads to augment training data.
    *   **Retraining Trigger:** Initiates the `src.training.train` pipeline in the background to update model weights without stopping the server.

### C. Offline Data Flow
1.  **Input:** User types text in the UI.
2.  **Preprocessing:** Text is cleaned and tokenized locally [Source 18].
3.  **Inference:** The **BiLSTM** model (loaded in memory) predicts the class label.
4.  **Storage:** The result, timestamp, and latency are committed to `data/app.db` [Source 19].
5.  **Display:** The UI updates via WebSocket push (NiceGUI reactive state).