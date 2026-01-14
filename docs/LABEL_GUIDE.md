# 🏷️ Label Guide & Annotation Schema

This project follows the strict labeling schema defined in **KRIXION Project Brief Section 4**.

## Class Definitions

| Label ID | Class Name | Description | UI Color [Source 21] |
|:--------:|:----------:|:------------|:--------------------:|
| **0** | **Normal** | Non-toxic, standard conversation. | 🟩 Green |
| **1** | **Offensive** | Rude, vulgar, or insulting language (but not hate speech). | 🟨 Amber |
| **2** | **Hate Speech** | Targeted attacks based on religion, caste, gender, etc. | 🟥 Red |

## Database Storage
In the SQLite database (`app.db`), predictions are stored in the `predicted_label` column as integers `(0, 1, 2)`.

## Annotation Rules
1. **Hierarchy:** If a text contains both Offensive and Hate terms, it is labeled as **Hate (2)** (Higher severity).
2. **Code-Mixing:** Labels apply regardless of whether the text is in Hindi, English, or Hinglish.