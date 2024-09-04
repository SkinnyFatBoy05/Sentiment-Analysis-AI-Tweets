# Sentiment Analysis of Tweets

This project analyzes the sentiment of tweets using Natural Language Processing (NLP) techniques. The analysis involves preprocessing, visualizing data, and building a sentiment classifier. The goal is to categorize tweets into Positive, Negative, or Neutral sentiments.

## Introduction

This project aims to perform sentiment analysis on a large set of tweets. The sentiment analysis is done using various NLP techniques such as word clouds, time series analysis, and sentiment classification using a logistic regression model.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SkinnyFatBoy05/sentiment-analysis.git
   ```
2. Navigate to the project directory:
   ```bash
   cd sentiment-analysis
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your dataset (`sentiment.csv`) in the project's root directory.
2. Run the Jupyter notebook `Sentiment.ipynb` to perform the analysis.

### Steps in the Notebook:

- **Data Loading:** Load the tweet dataset and explore the basic structure.
- **Preprocessing:** Handle missing data, convert date columns, and extract relevant features.
- **Visualization:** Generate word clouds, time series plots, and distribution charts for sentiment analysis.
- **Sentiment Classification:** Train a Logistic Regression model using TF-IDF vectorization to classify tweets into Positive, Negative, or Neutral categories.
- **Results Visualization:** Display the results of sentiment analysis using various plots.

## Features

- Word cloud visualization for prevalent words in tweets.
- Time series analysis of tweet volume over different periods.
- Sentiment classification using a logistic regression model.
- Comparative analysis of tweet sentiment across different months.
- Visual representation of sentiment distribution over time.

## Results

- Achieved a sentiment classification accuracy of **87.42%** using Logistic Regression.
- Visualized sentiment trends over time, providing insights into how sentiment fluctuates.
- Displayed prevalent words in Positive, Negative, and Neutral tweets.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss potential changes.
