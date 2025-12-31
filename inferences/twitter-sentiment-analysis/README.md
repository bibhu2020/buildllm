---
title: Tweet Sentiment Analyzer
emoji: 📝
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: true
---

# Tweet Sentiment Analyzer

This is a **Streamlit app** that uses a fine-tuned BERT model to classify the sentiment of tweets.  
It predicts whether a tweet expresses **joy, sadness, anger, love, fear, or surprise**.

---

## How to Use

1. Enter one or multiple tweets in the text area (one per line).  
2. Click **Analyze Sentiment**.  
3. The app will display each tweet's predicted sentiment along with the confidence score.

---

## Example

Tweet: I love spending time with my friends!
Sentiment: joy (0.95)

Tweet: I feel so sad about the news today.
Sentiment: sadness (0.88)

---

## Model Used

- **Hugging Face Model:** `mishrabp/bert-base-uncased-tweet-sentiment-analysis`
- **Base Model:** `bert-base-uncased`
- **Task:** Tweet Sentiment Analysis
- **Language:** English

---

## About

This app is useful for:

- Social media managers to monitor audience sentiment.
- Researchers analyzing public reactions.
- Companies tracking customer feedback.
- Quick, automated sentiment analysis pipelines.

---

## Installation (for local run)

```bash
pip install streamlit transformers torch
streamlit run app.py
```

## Author

Developed by Bibhu Mishra

---
