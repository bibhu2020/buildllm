import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# --- Page config ---
st.set_page_config(
    page_title="Tweet Sentiment Analyzer",
    page_icon="📝",
    layout="centered"
)

# --- App title and description ---
st.title("📝 Tweet Sentiment Analyzer")
st.markdown("""
This app uses a fine-tuned BERT model (**_mishrabp/bert-base-uncased-tweet-sentiment-analysis_**) to classify the sentiment of tweets.
You can enter one or multiple tweets (one per line), and the model will predict 
whether each tweet expresses **joy, sadness, anger, love, fear, or surprise**.
""")

# --- Load model and tokenizer ---
@st.cache_resource(show_spinner=True)
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return classifier

model_name = "mishrabp/bert-base-uncased-tweet-sentiment-analysis"
classifier = load_model(model_name)

# --- Text input for tweets ---
tweets_input = st.text_area(
    "Enter your tweets (one per line):",
    height=200
)

# --- Button to run classification ---
if st.button("Analyze Sentiment"):
    if not tweets_input.strip():
        st.warning("Please enter at least one tweet to analyze.")
    else:
        tweets = [line.strip() for line in tweets_input.split("\n") if line.strip()]
        with st.spinner("Classifying tweets..."):
            results = classifier(tweets)

        # --- Display results ---
        st.subheader("Results:")
        for tweet, result in zip(tweets, results):
            label = result["label"]
            score = result["score"]
            st.write(f"**Tweet:** {tweet}")
            st.write(f"**Sentiment:** {label} ({score:.2f})")
            st.markdown("---")
