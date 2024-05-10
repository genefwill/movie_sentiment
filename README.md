# Movie Review Sentiment Analyzer

This is a simple GUI application for analyzing the sentiment of movie reviews using a Support Vector Machine (SVM) model trained on the IMDb dataset.

## Instructions

1. Make sure you have Python installed on your system.

2. Clone or download this repository to your local machine.

3. Install the required dependencies using pip:

    ```
    pip install -r requirements.txt
    ```

4. Download the spacy model using:

    ```
    python -m spacy download en_core_web_lg
    ```
        

4. Run the application:

    ```
    python front_end.py
    ```

5. Enter your movie review in the provided text area and click the "Analyze" button to predict its sentiment.

## Accuracy

The SVM model used in this application achieves an accuracy of 88.52% on the IMDb dataset.