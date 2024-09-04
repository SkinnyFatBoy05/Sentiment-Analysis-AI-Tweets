# Sentiment Analysis on AI Tweets

## Description

This project performs sentiment analysis on a large dataset of tweets related to AI. The primary objective is to classify tweets into positive, negative, and neutral sentiments using various text processing and machine learning techniques.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/SkinnyFatBoy05/Sentiment-Analysis-AI-Tweets.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Sentiment-Analysis-AI-Tweets
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Import the necessary libraries:
    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wordcloud import WordCloud, STOPWORDS
    from nltk.sentiment import SentimentIntensityAnalyzer
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    ```

2. Load and explore the dataset:
    ```python
    df = pd.read_csv('sentiment.csv')
    df.head()
    ```

3. Perform data preprocessing:
    - Handle missing values.
    - Convert the 'Date' column to datetime format.
    - Extract features like year, month, day, etc.

4. Visualize the data:
    - Generate word clouds for tweets with different sentiments.
    - Plot time variation of tweet counts.
    - Analyze sentiment distribution over time.

5. Sentiment Analysis:
    - Use the VADER sentiment analyzer to classify tweets.
    - Train a Logistic Regression model using TF-IDF features for sentiment classification.
    - Evaluate the model's performance.

6. Visualize Results:
    - Plot the sentiment distribution as pie charts and histograms.
    - Analyze trends in tweet sentiments over time.

## Dataset

- The dataset used for this project consists of a large collection of tweets related to AI. It includes columns for the tweet text, date, user, and other metadata.
- Missing values were handled, and data was preprocessed for better analysis.

## Results

- The sentiment analysis model achieved an accuracy of 87.42% using Logistic Regression with TF-IDF features.
- Visualizations such as word clouds and time-series plots provided insights into the prevalent words and sentiment trends over time.

## Contributions

Feel free to contribute to this project by submitting a pull request. Any improvements or additional features are welcome!
