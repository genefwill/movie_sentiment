import tkinter as tk
from tkinter import scrolledtext, messagebox
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
import spacy
import re
import pickle

# Load English language model from spaCy
nlp = spacy.load('en_core_web_lg')

# Tokenize stopwords using spaCy
stopwords = nlp.Defaults.stop_words

# Load the trained model and TF-IDF vectorizer
with open('svm_model.pkl', 'rb') as f:
    svm = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)
with open('feature_selector.pkl', 'rb') as f:
    selector = pickle.load(f)

# Function to preprocess the input review
def preprocess_review(review):
    # remove any non words from text
    review = re.sub(r'[^a-zA-Z ]', '')
    #tokenize the review with spacy model
    tokens = nlp(review)
    # lemmatize the tokens
    nsw_tokens = [token.lemma_ for token in tokens if not token.text in stopwords]
    return " ".join(nsw_tokens)

# Function to categorize the review
def categorize_review():
    # Get entry from input
    review = review_entry.get("1.0",'end-1c')
    # Make sure review is entered, if not show message
    if not review:
        messagebox.showinfo("Error", "Please enter a review.")
        return
    # preprocess the review
    processed_review = preprocess_review(review)
    # transform review with tf-idf
    X = tfidf.transform([processed_review])
    # Select relevant features
    X_selected = selector.transform(X)
    # predict the sentiment with the trained SVM model
    category = svm.predict(X_selected)[0]
    # Displat the prediction
    if category == 1:
        category_label.config(text="Positive")
    else:
        category_label.config(text="Negative")

# Create the main window
window = tk.Tk()
window.geometry("500x350")
window.title("Movie Review Sentiment Analyzer")
window.configure(bg='#093f4d')

# Create the review input field
review_label = tk.Label(window, text="Enter your movie review:", bg='#093f4d', fg='white', font=("Arial", 14)) 
review_label.pack(pady=10)
review_entry = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=50, height=10, font=("Arial", 10))
review_entry.pack(pady=10, padx=20)

# Create the button to analyze the review
analyze_button = tk.Button(window, text="Analyze", command=categorize_review, bg='#8ff7ef', fg='black', font=("Arial", 12)) 
analyze_button.pack(pady=10)

# Create the label to display the category
category_label = tk.Label(window, text="", bg='#093f4d', fg='white', font=("Arial", 14)) 
category_label.pack(pady=10)

window.mainloop()