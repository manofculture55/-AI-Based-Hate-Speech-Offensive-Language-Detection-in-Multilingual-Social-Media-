# 📊 Data Sources & Preprocessing Report

## 1. Dataset Sources
This project integrates the following officially approved KRIXION datasets [4, 5]:

| Dataset Name | Original Format | Role in Project | Description |
| :--- | :--- | :--- | :--- |
| **Bohra et al. (2018)** | CSV/TSV | **Primary Training** | Gold-standard for Hindi-English Code-Mixed (Hinglish) hate speech. |
| **Indo-HateSpeech (2024)** | XLSX | **Supplementary** | Modern social media comments with heavy transliteration. |
| **HASOC 2019 (Hindi)** | TSV | **Validation** | Pure Hindi (Devanagari) and mixed script content. |
| **HASOC 2019 (English)** | TSV | **Validation** | Standard English hate speech baseline. |
| **MDPI 2025** | CSV | **Benchmarking** | Trilingual corpus used for final metric comparison. |

## 2. Preprocessing Pipeline
Following the mandatory pipeline defined in **Section 4** [6], the following steps were applied:

1.  **Text Cleaning:** Removed URLs, user mentions (`@user`), hashtags (`#topic`), and non-alphanumeric special characters.
2.  **Mojibake Fix:** Corrected encoding errors (Windows-1252 vs UTF-8) in the MDPI dataset.
3.  **Language Identification:** Implemented strict logic [6]:
    *   **Hindi (`hi`):** Text containing Devanagari script (e.g., "नमस्ते").
    *   **Hinglish (`hi-en`):** Latin script text originating from Code-Mixed sources (e.g., "Tum pagal ho").
    *   **English (`en`):** Latin script text originating from English-only sources.
4.  **Label Mapping:** Standardized all source labels to the KRIXION schema [3]:
    *   **0 (Normal):** Non-toxic content (mapped from `NOT`, `NONE`, `HS0`).
    *   **1 (Offensive):** Abusive/Profane but not hateful (mapped from `OFFN`, `PRFN`, `HS1`).
    *   **2 (Hate Speech):** Targeted attacks (mapped from `HATE`, `HSN`).

## 3. Storage Architecture
*   **Raw Data:** Stored in `data/` as original CSV/XLSX files.
*   **Processed Data:** Unified into `data/clean_data.csv`.
*   **Database:** All samples loaded into SQLite `data/app.db` (Table: `annotations`) [7].