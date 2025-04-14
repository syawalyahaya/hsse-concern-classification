#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from datetime import datetime

# Download English stopwords
nltk.download("stopwords")
english_stopwords = set(stopwords.words("english"))

# Get Malay stopwords using Sastrawi
factory = StopWordRemoverFactory()
malay_stopwords = set(factory.get_stop_words())

# Combine both sets of stopwords
all_stopwords = english_stopwords.union(malay_stopwords)

# Define file paths (replace with your actual file locations)
raw_file = 'path/to/HSE_Concern_Form.xlsx'
prev_classified_file = 'path/to/Python_Concern_Classification_YYYY-MM-DD.xlsx'

# Define the column containing the concern descriptions
column_name = 'Description of concern'

# Load previous classified entries
prev_df = pd.read_excel(prev_classified_file)
prev_sentences = set(prev_df['Original Sentence'].dropna().unique())

# Load new/raw concern data
concern_df = pd.read_excel(raw_file)
concern_df["Original Sentence"] = concern_df.get(column_name)

# Filter only new/unseen concerns
new_df = concern_df[~concern_df["Original Sentence"].isin(prev_sentences)].copy()

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    words = [w for w in text.split() if w not in all_stopwords]
    return " ".join(words)

# Apply cleaning
new_df["Cleaned Sentence"] = new_df["Original Sentence"].apply(clean_text)

# Predefined keyword categories
slippery_keywords = {
    "slippery", "lantai", "licin", "minyak", "tumpah", "water", "leakage",
    "basah", "watery", "leak", "leaking", "bocor", "leaks", "bertakung",
    "bertakong", "air", "wet", "floor"
}
test_keywords = {"test", "testing", "abc", "try", "trial", "tetsting", "x", "huhu", "welcome"}
animal_keywords = {
    "cat", "kucing", "tikus", "rat", "mouse", "dog", "anjing", "animal",
    "binatang", "serangga", "tebuan", "lebah", "sarang", "anai-anai", "agas",
    "liar", "burung", "berbisa", "bisa", "venomous", "venom", "poisonous",
    "musang", "bangkai", "haiwan", "cats", "dogs", "ular", "snakes", "snake",
    "nyamuk", "jejentik"
}

# Assign to predefined cluster
def assign_predefined_cluster(text):
    if not isinstance(text, str) or text.strip() == "" or re.fullmatch(r"[a-zA-Z]", text.strip()):
        return 1  # Dummy/testing entry
    words = set(text.split())
    if words & slippery_keywords:
        return 0  # Slippery/wet area
    elif words & test_keywords:
        return 1  # Test
    elif words & animal_keywords:
        return 2  # Animal/pest
    return -1  # Unknown/unassigned

new_df["Predefined Cluster"] = new_df["Cleaned Sentence"].apply(assign_predefined_cluster)

# Separate assigned and unassigned entries
assigned_df = new_df[new_df["Predefined Cluster"] != -1].copy()
unassigned_df = new_df[new_df["Predefined Cluster"] == -1].copy()

# Apply ML clustering to unassigned entries
if not unassigned_df.empty:
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(unassigned_df["Cleaned Sentence"])
    kmeans = KMeans(n_clusters=5, random_state=42)
    unassigned_df["Cluster"] = kmeans.fit_predict(X)
    cluster_counts = Counter(unassigned_df["Cluster"])
    top_clusters = [c[0] for c in cluster_counts.most_common(5)]
    unassigned_df["Segment"] = unassigned_df["Cluster"].apply(lambda x: x if x in top_clusters else -1)
else:
    unassigned_df["Segment"] = []

# Use predefined cluster value as segment
assigned_df["Segment"] = assigned_df["Predefined Cluster"]

# Combine both sets
classified_new_df = pd.concat([assigned_df, unassigned_df], ignore_index=True)

# Add human-readable cluster names
cluster_name_map = {
    0: "Slippery / Wet Area",
    1: "Testing / Dummy Text",
    2: "Animal / Pest",
    3: "Cluster 3",
    4: "Cluster 4",
    -1: "Others / Unassigned"
}
classified_new_df["Cluster Name"] = classified_new_df["Segment"].map(cluster_name_map).fillna("Others / Unassigned")

# Merge with previous data
final_df = pd.concat([prev_df, classified_new_df], ignore_index=True)

# Export to Excel
today = datetime.today().strftime('%Y-%m-%d')
output_file = f'output/Python_Concern_Classification_{today}.xlsx'
final_df.to_excel(output_file, index=False)

print(f"✅ Classification complete. File saved: {output_file}")

