# Label Guide & Annotation Schema

This project follows the strict labeling schema defined in **KRIXION Project Brief Section 4** [Source 21].

## 1. Class Definitions
| Label ID | Class Name | UI Color | Description |
|:---:|:---:|:---:|:---|
| **0** | **Normal** | 🟩 Green | Non-toxic, standard conversation. No aggression detected. |
| **1** | **Offensive** | 🟨 Amber | Rude, vulgar, or insulting language (but NOT hate speech). |
| **2** | **Hate Speech** | 🟥 Red | Targeted attacks based on religion, caste, gender, or ethnicity. |

## 2. Database Storage
In the SQLite database (`app.db`), predictions are stored in the `predicted_label` column as integers `(0, 1, 2)`.

## 3. Annotation Rules
1.  **Hierarchy Rule:** If a text contains both Offensive terms (e.g., swear words) and Hate Speech (e.g., slurs against a group), it is labeled as **Hate (2)** because it has higher severity.
2.  **Code-Mixing Rule:** Labels apply regardless of language. A hate speech comment in Hinglish (e.g., "Tum [slur] ho") is treated exactly the same as in English or Hindi.

--------------------------------------------------------------------------------