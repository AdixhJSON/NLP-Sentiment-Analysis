# ============================================================
# NLP-BASED CUSTOMER REVIEW SENTIMENT ANALYSIS
# STREAMLIT APPLICATION
# ============================================================


# 1. IMPORT REQUIRED LIBRARIES

import streamlit as st
import pandas as pd
import re
import joblib
import os


# 2. PAGE CONFIGURATION

st.set_page_config(
    page_title="Customer Review Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)


# 3. MODEL FILE PATHS

MODEL_PATH = "svm_sentiment_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"


# 4. LOAD TRAINED SVM MODEL AND TF-IDF VECTORIZER

@st.cache_resource
def load_model(model_path, vectorizer_path, model_modified, vectorizer_modified):

    model = joblib.load(model_path)

    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer


# Check whether model files exist

if not os.path.exists(MODEL_PATH):

    st.error("SVM model file not found!")

    st.stop()


if not os.path.exists(VECTORIZER_PATH):

    st.error("TF-IDF vectorizer file not found!")

    st.stop()


# Load the saved model

try:

    model, tfidf = load_model(
        MODEL_PATH,
        VECTORIZER_PATH,
        os.path.getmtime(MODEL_PATH),
        os.path.getmtime(VECTORIZER_PATH)
    )

    st.success("Improved SVM model loaded successfully!")

except Exception as e:

    st.error(f"Error loading model files: {e}")

    st.stop()


# 5. TEXT CLEANING FUNCTION

def clean_text(text):

    # Convert text into lowercase
    text = str(text).lower()

    # Remove website URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters, numbers and punctuation
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# 6. SENTIMENT PREDICTION FUNCTION

def predict_sentiment(reviews):

    # Clean customer reviews
    cleaned_reviews = [
        clean_text(review) for review in reviews
    ]

    # Convert cleaned reviews into TF-IDF features
    review_vectors = tfidf.transform(cleaned_reviews)

    # Predict sentiment using improved SVM
    predictions = model.predict(review_vectors)

    return predictions, review_vectors


# 7. APPLICATION HEADER

st.title("📊 Customer Review Sentiment Analysis")

st.markdown(
    """
    This application uses **Natural Language Processing (NLP)**
    and **Machine Learning** to analyse customer product reviews.

    It predicts whether a customer review expresses
    **Positive, Negative or Neutral sentiment.**
    """
)


st.info(
    """
    **Dataset:** Samsung Mobile Phone Customer Reviews

    **Machine Learning Algorithm:** Improved Support Vector Machine (SVM)

    **Feature Extraction:** TF-IDF
    """
)


# 8. MODEL INFORMATION

st.header("Model Information")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        label="Machine Learning Model",
        value="Improved SVM"
    )


with col2:

    st.metric(
        label="Feature Extraction",
        value="TF-IDF"
    )


st.divider()


# ============================================================
# SECTION 1: SINGLE CUSTOMER REVIEW ANALYSIS
# ============================================================

st.header("1. Analyse a Customer Review")

st.write(
    "Enter a customer review below to predict its sentiment."
)


# Customer review input

user_review = st.text_area(
    "Enter Customer Review:",
    placeholder="Example: The phone has excellent battery life and a great camera.",
    height=150
)


# Prediction button

if st.button("Analyse Sentiment", type="primary"):

    if user_review.strip() == "":

        st.warning("Please enter a customer review first.")

    elif clean_text(user_review) == "":

        st.warning("The review does not contain usable English text.")

    else:

        # Predict sentiment
        predictions, review_vectors = predict_sentiment(
            [user_review]
        )

        sentiment = predictions[0]

        # Display prediction
        st.subheader("Sentiment Prediction")

        if sentiment == "Positive":

            st.success("😊 Positive Sentiment")

        elif sentiment == "Negative":

            st.error("😟 Negative Sentiment")

        elif sentiment == "Neutral":

            st.info("😐 Neutral Sentiment")


        st.write("Predicted Sentiment:", sentiment)


        # Display SVM decision scores

        if hasattr(model, "decision_function"):

            decision_scores = model.decision_function(
                review_vectors
            )

            if len(model.classes_) == 3:

                score_df = pd.DataFrame({

                    "Sentiment": model.classes_,

                    "Decision Score": decision_scores[0]

                })

                st.subheader("SVM Sentiment Classification Scores")

                st.bar_chart(
                    score_df.set_index("Sentiment")
                )

                st.caption(
                    "Each bar represents the SVM decision score "
                    "for a sentiment category. A higher score "
                    "indicates a stronger preference for that "
                    "category. These scores are not probabilities."
                )


st.divider()


# ============================================================
# SECTION 2: MULTIPLE CUSTOMER REVIEW ANALYSIS
# ============================================================

st.header("2. Analyse Multiple Customer Reviews")

st.write(
    """
    Upload a CSV file containing customer reviews.
    The application will predict the sentiment of each review.
    """
)


# Upload CSV file

uploaded_file = st.file_uploader(
    "Upload Customer Reviews CSV File",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        # Read uploaded CSV file
        uploaded_df = pd.read_csv(uploaded_file)

        # Remove extra spaces from column names
        uploaded_df.columns = uploaded_df.columns.str.strip()

        # Convert column names to lowercase
        uploaded_df.columns = uploaded_df.columns.str.lower()


        # Identify review column

        if "review" in uploaded_df.columns:

            review_column = "review"

        elif "body" in uploaded_df.columns:

            review_column = "body"

        else:

            st.error(
                "CSV file must contain a column named "
                "'review' or 'body'."
            )

            st.stop()


        # Remove empty reviews

        uploaded_df = uploaded_df.dropna(
            subset=[review_column]
        ).copy()

        uploaded_df[review_column] = (
            uploaded_df[review_column].astype(str)
        )

        uploaded_df = uploaded_df[
            uploaded_df[review_column].str.strip() != ""
        ].copy()


        if uploaded_df.empty:

            st.warning("No valid customer reviews found.")

            st.stop()


        # Display uploaded dataset

        st.subheader("Uploaded Dataset")

        st.dataframe(
            uploaded_df.head(),
            use_container_width=True
        )


        # Analyse button

        if st.button("Analyse All Reviews"):

            # Predict sentiment
            predictions, review_vectors = predict_sentiment(
                uploaded_df[review_column].tolist()
            )


            # Add predictions to dataset

            results_df = uploaded_df.copy()

            results_df["Predicted Sentiment"] = predictions


            # Display results

            st.subheader("Sentiment Analysis Results")

            st.dataframe(
                results_df,
                use_container_width=True
            )


            # ====================================================
            # SENTIMENT DISTRIBUTION
            # ====================================================

            st.subheader("Sentiment Distribution")


            sentiment_order = [
                "Negative",
                "Neutral",
                "Positive"
            ]


            sentiment_counts = (
                results_df["Predicted Sentiment"]
                .value_counts()
                .reindex(
                    sentiment_order,
                    fill_value=0
                )
            )


            # Create graph

            chart_df = pd.DataFrame({

                "Number of Reviews": sentiment_counts

            })


            st.bar_chart(chart_df)


            # ====================================================
            # PREDICTION SUMMARY
            # ====================================================

            st.subheader("Prediction Summary")


            positive_count = int(
                sentiment_counts["Positive"]
            )

            negative_count = int(
                sentiment_counts["Negative"]
            )

            neutral_count = int(
                sentiment_counts["Neutral"]
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.metric(
                    "Positive Reviews",
                    positive_count
                )


            with col2:

                st.metric(
                    "Negative Reviews",
                    negative_count
                )


            with col3:

                st.metric(
                    "Neutral Reviews",
                    neutral_count
                )


            st.metric(
                "Total Reviews Analysed",
                len(results_df)
            )


            # ====================================================
            # DOWNLOAD RESULTS
            # ====================================================

            csv_data = results_df.to_csv(
                index=False
            ).encode("utf-8")


            st.download_button(

                label="Download Sentiment Results",

                data=csv_data,

                file_name="sentiment_analysis_results.csv",

                mime="text/csv"

            )


    except pd.errors.EmptyDataError:

        st.error("The uploaded CSV file is empty.")


    except Exception as e:

        st.error(
            f"An error occurred while processing the file: {e}"
        )


# ============================================================
# APPLICATION FOOTER
# ============================================================

st.divider()

st.caption(
    "NLP-Based Customer Review Sentiment Analysis Project"
)

st.caption(
    "Developed using Python, Natural Language Processing, "
    "TF-IDF, Improved Support Vector Machine and Streamlit."
)