# hsse-concern-classification

# 🧠 HSE Concern Classifier

This repository contains a Python script that automatically classifies health, safety, and environment (HSE) concerns using both keyword rules and machine learning (KMeans clustering). It's designed to help safety teams quickly analyze and group textual concern reports for more efficient review and action.

---

## 📌 Use Case

Every month, safety concerns are collected from staff across various sites. These come in the form of free-text descriptions that vary in language (Malay & English) and structure.

Manually reviewing hundreds of entries is time-consuming — so we automated it.

This script:
- Identifies new, unprocessed concern entries
- Cleans and processes the text
- Tags familiar patterns like:
  - Slippery/wet areas
  - Test/dummy entries
  - Animal/pest sightings
- Uses **TF-IDF** and **KMeans** to group unknown patterns
- Saves the results with human-readable cluster names

---

## 🧰 Tech Stack

- `pandas` for data manipulation
- `nltk` & `Sastrawi` for English and Malay stopword handling
- `scikit-learn` for:
  - TF-IDF vectorization
  - KMeans clustering
- `openpyxl` or `xlsxwriter` for Excel I/O

---

## 🔄 How It Works

1. **Load previous data:** Ensures we don’t double-process older entries.
2. **Read new raw entries** from the latest Excel concern form.
3. **Preprocessing:**
   - Case normalization
   - Punctuation removal
   - Stopword filtering (English + Malay)
4. **Categorize:**
   - Use keyword sets to assign known issues (e.g., `slippery`, `kucing`, `test`)
   - Use KMeans to cluster everything else
5. **Add human-readable labels** like:
   - Slippery / Wet Area
   - Animal / Pest
   - Cluster 3, 4, ...
6. **Combine and export** all entries (old + new) to a fresh Excel file with today's date.
