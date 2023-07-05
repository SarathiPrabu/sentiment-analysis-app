import streamlit as st
from transformers import pipeline

# Create a sentiment analysis pipeline using a specific model
nlp_pipeline = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Streamlit application title
st.title("Sentiment Analysis")

# Text input box
text_input = st.text_area("Enter text for sentiment analysis")

# Analyze button
if st.button("Analyze"):
    if text_input:
        # Perform sentiment analysis on the input text
        sentiment = nlp_pipeline(text_input)

        # Get the sentiment label and score
        label = sentiment[0]["label"]
        score = sentiment[0]["score"]

        # Display the sentiment analysis result
        st.write("Sentiment:", label)
        st.write("Confidence:", score)

        # Map sentiment label to positive, negative, or neutral
        sentiment_mapping = {
            "NEGATIVE": "Negative",
            "NEUTRAL": "Neutral",
            "POSITIVE": "Positive",
        }
        mapped_sentiment = sentiment_mapping.get(label)

        # Display the mapped sentiment
        st.write("Mapped Sentiment:", mapped_sentiment)
    else:
        st.warning("Please enter some text for analysis.")
