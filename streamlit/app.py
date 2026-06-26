import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Amazon Reviews Sentiment Analysis Dashboard",
    layout="wide"
)

# Image function
def show_image(path, caption):
    if Path(path).exists():
        st.image(path, use_container_width=True, caption=caption)
    else:
        st.error(f"Missing file: {path}")

# Navbar
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Sentiment Distribution",
        "Transformer-Based Model Comparison",
        "Deep Learning Model Results",
        "Word Cloud Analysis",
        "Aspect-Based Sentiment Analysis",
        "Model Performance Comparison",
        "Sentiment Trend Over Time"
    ]
)

# Title
st.title("Amazon Reviews Sentiment Analysis Visualisation Dashboard")
st.write("Sentiment analysis using transformer-based models, deep learning models, and aspect-based sentiment analysis on Amazon reviews.")

st.divider()

# Sentiment Distribution
if page == "Sentiment Distribution":
    st.header("Sentiment Distribution")

    show_image("rating_distribution.png", "Rating Distribution in Amazon Reviews Dataset")

# Transformel model
elif page == "Transformer-Based Model Comparison":
    st.header("Transformer-Based Model Comparison")

    col1, col2, col3 = st.columns(3)

    with col1:
        show_image("roberta-base_confusion_matrix.png", "RoBERTa Base Model — Confusion Matrix")
        show_image("roberta-base_training_curves.png", "RoBERTa Base Model — Training Curves")

    with col2:
        show_image("distilbert-base-uncased_confusion_matrix.png", "DistilBERT Base Uncased — Confusion Matrix")
        show_image("distilbert-base-uncased_training_curves.png", "DistilBERT Base Uncased — Training Curves")

    with col3:
        show_image("bert-base-uncased_confusion_matrix.png", "BERT Base Uncased — Confusion Matrix")
        show_image("bert-base-uncased_training_curves.png", "BERT Base Uncased — Training Curves")

# Deep Learning
elif page == "Deep Learning Model Results":
    st.header("Deep Learning Model Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Long Short-Term Memory Model")

        show_image("lstm_confusion_matrix.png", "Long Short-Term Memory — Confusion Matrix" )
        show_image("lstm_curves.png", "Long Short-Term Memory — Training Curves")

    with col2:
        st.subheader("Bidirectional Long Short-Term Memory Model")

        show_image("bilstm_confusion_matrix.png", "Bidirectional Long Short-Term Memory — Confusion Matrix")
        show_image("bilstm_curves.png", "Bidirectional Long Short-Term Memory — Training Curves")

# Word Clouds
elif page == "Word Cloud Analysis":

    st.header("Word Cloud Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        show_image("positive_reviews_word_cloud_wordcloud.png", "Positive Reviews")

    with col2:
        show_image("neutral_reviews_word_cloud_wordcloud.png", "Neutral Reviews")

    with col3:
        show_image("negative_reviews_word_cloud_wordcloud.png", "Negative Reviews")

# ABSA
elif page == "Aspect-Based Sentiment Analysis":
    st.header("Aspect-Based Sentiment Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        show_image("absa_aspect_sentiment_bar.png", "Aspect Sentiment Distribution")

    with col2:
        show_image("absa_aspect_pct.png", "Aspect Sentiment Percentage Breakdown")

    with col3:
        show_image("absa_aspect_heatmap.png", "Aspect Sentiment Heatmap")

# Performance Comparison
elif page == "Model Performance Comparison":
    st.header("Model Performance Comparison")
    show_image("all_models_heatmap.png", "Model Performance Heatmap")

# Trend
elif page == "Sentiment Trend Over Time":
    st.header("Sentiment Trend Over Time")

    show_image("monthly_sentiment_trend.png", "Monthly Sentiment Trend")