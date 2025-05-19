import streamlit as st
import joblib
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# Ensure NLTK stopwords are available
nltk.download("stopwords", download_dir="nltk_data")
nltk.data.path.append("nltk_data")  # Set custom directory for NLTK

# Load trained model & vectorizer
svm_model = joblib.load("svm_model.pkl")
cv = joblib.load("vectorizer.pkl")

# Initialize text preprocessing tools
ps = PorterStemmer()
custom_stopwords = {
    'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'no', 'not', 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}
stop_words = set(stopwords.words("english")) - custom_stopwords

# Text preprocessing function
def preprocess_text(review_text):
    review = re.sub('[^a-zA-Z]', ' ', review_text)  # Remove non-letters
    review = review.lower()  # Convert to lowercase
    review = review.split()  # Tokenize
    review = [ps.stem(word) for word in review if word not in stop_words]  # Remove stopwords & stem words
    review = " ".join(review)  # Rejoin words
    
    return cv.transform([review]).toarray()  # Convert processed text to numerical format

# Streamlit UI
st.title("Customer Review Classifier")
st.write("Enter a customer review below to predict whether it's Positive or Negative.")

# User input
review_input = st.text_area("Enter your review:")

# Prediction function
def predict_review(review_text):
    processed_text = preprocess_text(review_text)  # Apply preprocessing
    prediction = svm_model.predict(processed_text)
    return "Positive" if prediction[0] == 1 else "Negative"

# Classify review button
if st.button("Classify Review"):
    if review_input.strip():
        result = predict_review(review_input)
        st.success(f"Prediction: **{result}**")
    else:
        st.warning("Please enter a review before classifying.")

# Run the app with: python -m streamlit run customer_review.py
