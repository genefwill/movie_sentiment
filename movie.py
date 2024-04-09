
import numpy as np
import pandas as pd
import os
import random



def load_imdb_sentiment_analysis_dataset():

    training_data = pd.read_csv("data\\imdb_dataset.csv")

    texts = []
    cleaned_data = training_data[["review", "sentiment"]]
    print(len(cleaned_data))
    
    



load_imdb_sentiment_analysis_dataset()
    