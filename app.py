import streamlit as st
from transformers import pipeline
import torch
import random

# Check if GPU is available
if torch.cuda.is_available():
    device = 0
else:
    # If GPU is not available, use CPU
    device = torch.device("cpu")

model_name = "facebook/bart-large-mnli"
nlp_pipeline = pipeline("zero-shot-classification", model=model_name, device=device)

# Random text for pre population. If we want to enter custom texts, this can be commented
random_reviews = [
    "This movie was fantastic! I loved every minute of it.",
    "The food at this restaurant is awful. I would not recommend it.",
    "I had a great experience at this hotel. The staff was friendly and helpful.",
    "The service was slow and the prices were too high. I was disappointed.",
    "I absolutely adore this book. It's a must-read for everyone.",
]

# Streamlit application title
st.title("Sentiment Analysis - Hugging Space")

# Text input box
text_input = st.text_area(
    "Enter text for sentiment analysis", value=random.choice(random_reviews)
)
text_labels = ["Positive", "Negative", "Neutral"]

# Analyze button
if st.button("Analyze"):
    if text_input:
        results = nlp_pipeline(text_input, text_labels)
        sentiment = results["labels"][0]
        confidence = results["scores"][0]

        # Display the sentiment analysis results
        st.write("Sentiment:", sentiment)
        st.write("Confidence:", round(confidence * 100, 2), "%")

    else:
        st.warning("Please enter some text for analysis.")
