import numpy as np
import pandas as pd
import os
import random
import spacy
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

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

# Convert labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Loading the TF-IDF vectorizer and feature selector pickles
with open('tfidf+vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open ('feature_selector.pkl', 'rb') as f:
    selector = pickle.load(f)

# Convert text data into TF-IDF vectors with feature selection
tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
X = tfidf.fit_transform(docs)

# Perform feature selection using chi-squared test
selector = SelectKBest(chi2, k=min(2000, X.shape[1]))  # Select top 2000 features
X_selected = selector.fit_transform(X, y_encoded)

# pickle for trained model
with open('trained_model.pkl', 'rb') as f:
    mnb = pickle.load(f)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y_encoded, test_size=0.5, random_state=0)

# Initialize Multinomial Naive Bayes classifier
mnb = MultinomialNB()

# Train the classifier and predict labels for testing set
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)

# Calculate the accuracy
accuracy = (y_pred == y_test).mean()
print("Accuracy:", accuracy)
