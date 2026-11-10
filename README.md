# NLP Customer Review Sentiment Analysis

A three-class NLP project that classifies Samsung mobile phone reviews as positive, neutral, or negative.

## Method

- Cleaned review text and combined review titles with review bodies.
- Converted text to TF-IDF features using up to 5,000 unigram and bigram features.
- Compared Multinomial Naive Bayes, Logistic Regression, and LinearSVC.
- Selected a class-balanced LinearSVC model.

## Results

On a stratified test split of 288 reviews, the selected model achieved **81.94% accuracy** and **69.23% macro F1**.

## Streamlit app

The app supports individual review predictions and CSV batch analysis, displays sentiment summaries, and lets users download results.

To run the app, install the required packages:

    pip install streamlit pandas scikit-learn joblib

Then, from the project folder, run:

    streamlit run app.py

Keep `svm_sentiment_model.pkl` and `tfidf_vectorizer.pkl` in the same folder as `app.py`.

## Dataset

The notebook expects a file named `Dataset (2).xlsx`, which is not included in this repository. Add it locally to rerun the notebook; share it publicly only if you have permission.

## Limitation

Text preprocessing is designed primarily for English reviews. Neutral sentiment remains harder for the model to identify than positive or negative sentiment.
