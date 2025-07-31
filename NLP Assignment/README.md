# Movie Review Sentiment Analysis

This project implements a full text classification pipeline using the IMDb movie review dataset. It includes text preprocessing, feature engineering (BoW, TF-IDF, Word2Vec), and an optional trigram-based Markov model for text generation.

## Features

- Text preprocessing: cleaning, tokenization, lemmatization
- Feature engineering: CountVectorizer, TF-IDF, Word2Vec
- Optional: Markov Chain generator (trigram-based)
- Optional: Sentiment classification with Logistic Regression

## Getting Started

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook: `text_classification_pipeline.ipynb`

## Dataset

IMDb Movie Review Dataset (binary sentiment classification)

## Output

- Cleaned dataset
- Vectorized features
- Generated text samples
- Evaluation metrics (if classification is included)
