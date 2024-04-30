import numpy as np
import pandas as pd
import os
import random
import spacy
import re


# Load English language model from spaCy
nlp = spacy.load('en_core_web_lg')

# Load IMDb dataset
data = pd.read_csv("data/imdb_dataset.csv")
cleaned_data = data[["review", "sentiment"]]

# Tokenize stopwords using spaCy
stopwords = nlp.Defaults.stop_words

# Initialize lists to store processed data
docs = []
y = []

# Process each row in the dataset. cleaning data
# remove stop words and html tags
for index, row in cleaned_data.iterrows():
    review = row["review"]
    review = re.sub(r'[^a-zA-Z ]', '', review)
    tokens = nlp(review)
    sentiment = row["sentiment"]
    gen = row["sentiment"]
    if (110 < len(tokens) <= 500):
        y.append(gen)
        nsw_tokens = [token.lemma_ for token in tokens if not token.text in stopwords]
        docs.append(" ".join(nsw_tokens))


