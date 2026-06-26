import streamlit as st
import pandas as pd
from pathlib import Path

# Page setup
st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    layout="wide"
)

# Folder paths
BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"

st.title("AI-Based Sentiment Analysis on Amazon Product Reviews")

st.write(
    "This dashboard presents the main results of the sentiment analysis project, "
    "including sentiment distribution, word clouds, model comparison, confusion matrix, "
    "and aspect-based sentiment analysis."
)

# Sidebar menu
menu = st.sidebar.selectbox(
    "Select Section",
    [
        "Project Overview",
        "Dataset",
        "Sentiment Distribution",
        "Word Clouds",
        "Model Comparison",
        "Confusion Matrix",
        "Aspect-Based Sentiment Analysis"
    ]
)

# Helper functions
def show_image(file_name, caption):
    file_path = OUTPUTS_DIR / file_name
    if file_path.exists():
        st.image(str(file_path), caption=caption, use_container_width=True)
    else:
        st.warning(f"File not found: {file_name}")

def show_csv(file_name):
    file_path = OUTPUTS_DIR / file_name
    if file_path.exists():
        df = pd.read_csv(file_path)
        st.dataframe(df, use_container_width=True)
    else:
        st.warning(f"File not found: {file_name}")


# Pages
if menu == "Project Overview":
    st.header("Project Overview")
    st.write(
        "This project analyzes Amazon product reviews using Natural Language Processing. "
        "The reviews are classified into Negative, Neutral, and Positive sentiments. "
        "Several models were compared, including traditional machine learning, deep learning, "
        "and transformer-based models."
    )

elif menu == "Dataset":
    st.header("Dataset")

    cleaned_file = DATA_DIR / "Amazon_Reviews_Cleaned.csv"
    preprocessed_file = DATA_DIR / "Amazon_Reviews_Preprocessed.csv"

    if cleaned_file.exists():
        df = pd.read_csv(cleaned_file)
        st.subheader("Cleaned Dataset Preview")
        st.write(f"Number of rows: {df.shape[0]}")
        st.write(f"Number of columns: {df.shape[1]}")
        st.dataframe(df.head(), use_container_width=True)
    else:
        st.warning("Cleaned dataset file not found.")

    if preprocessed_file.exists():
        df_pre = pd.read_csv(preprocessed_file)
        st.subheader("Preprocessed Dataset Preview")
        st.dataframe(df_pre.head(), use_container_width=True)
    else:
        st.warning("Preprocessed dataset file not found.")

elif menu == "Sentiment Distribution":
    st.header("Sentiment Distribution")
    show_image("rating_distribution.png", "Rating Distribution")
    show_image("sentiment_distribution_pie.png", "Sentiment Class Distribution")

elif menu == "Word Clouds":
    st.header("Word Cloud Analysis")
    show_image("positive_reviews_word_cloud_wordcloud.png", "Word Cloud for Positive Reviews")
    show_image("negative_reviews_word_cloud_wordcloud.png", "Word Cloud for Negative Reviews")
    show_image("neutral_reviews_word_cloud_wordcloud.png", "Word Cloud for Neutral Reviews")

elif menu == "Model Comparison":
    st.header("Model Performance Comparison")
    show_csv("full_model_comparison.csv")
    show_image("model_comparison.png", "Model Comparison")

elif menu == "Confusion Matrix":
    st.header("RoBERTa Confusion Matrix")
    show_image("roberta-base_confusion_matrix.png", "RoBERTa Confusion Matrix")

elif menu == "Aspect-Based Sentiment Analysis":
    st.header("Aspect-Based Sentiment Analysis")
    show_image("absa_aspect_sentiment_bar.png", "Aspect-Wise Sentiment Distribution")
    show_image("absa_aspect_heatmap.png", "Aspect-Based Sentiment Heatmap")
