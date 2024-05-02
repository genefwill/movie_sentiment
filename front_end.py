import tkinter as tk
from tkinter import scrolledtext, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
import spacy
import re
import pickle

# Load English language model from spaCy
nlp = spacy.load('en_core_web_lg')

# Tokenize stopwords using spaCy
stopwords = nlp.Defaults.stop_words

# Load the trained model and TF-IDF vectorizer
mnb = MultinomialNB()
tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
selector = SelectKBest(chi2, k=2000)
with open('trained_model.pkl', 'rb') as f:
    mnb = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('feature_selector.pkl', 'rb') as f:
    selector = pickle.load(f)

# Function to preprocess the input review
def preprocess_review(review):
    review = re.sub(r'[^a-zA-Z ]', '', review)
    tokens = nlp(review)
    nsw_tokens = [token.lemma_ for token in tokens if not token.text in stopwords]
    return " ".join(nsw_tokens)

# Function to categorize the review
def categorize_review():
    review = review_entry.get("1.0",'end-1c')
    if not review:
        messagebox.showwarning("Warning", "Please enter a review.")
        return
    processed_review = preprocess_review(review)
    X = tfidf.transform([processed_review])
    X_selected = selector.transform(X)
    category = mnb.predict(X_selected)[0]
    if category == 1:
        category_label.config(text="Positive")
    else:
        category_label.config(text="Negative")

# Create the main window
window = tk.Tk()
window.title("Movie Review Sentiment Analyzer")

# Create the review input field
review_label = tk.Label(window, text="Enter your movie review:")
review_label.pack()
review_entry = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=50, height=10)
review_entry.pack()

# Create the button to analyze the review
analyze_button = tk.Button(window, text="Analyze", command=categorize_review)
analyze_button.pack()

# Create the label to display the category
category_label = tk.Label(window, text="")
category_label.pack()

window.mainloop()