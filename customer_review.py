import streamlit as st
import joblib
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Download stopwords to custom path
nltk.download("stopwords", download_dir="nltk_data")
nltk.data.path.append("nltk_data")

# Load model and vectorizer
svm_model = joblib.load("svm_model.pkl")
cv = joblib.load("vectorizer.pkl")

# Preprocessing tools
ps = PorterStemmer()
custom_stopwords = {
    'don', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't",
    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't",
    'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'no', 'not', 'shouldn', "shouldn't",
    'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}
stop_words = set(stopwords.words("english")) - custom_stopwords

# Preprocessing function
def preprocess_text(review_text):
    review = re.sub('[^a-zA-Z]', ' ', review_text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stop_words]
    return cv.transform([" ".join(review)]).toarray()

# 🔁 Prediction logic (MUST come before it's called)
def predict_review(text):
    processed_text = preprocess_text(text)
    prediction = svm_model.predict(processed_text)
    return "Positive" if prediction[0] == 1 else "Negative"

# Streamlit App UI
st.set_page_config(page_title="Restaurant Review Classifier", layout="centered")
st.title("🍽️ Restaurant Review Classifier")
st.write("Enter a restaurant review to classify it as **Positive** or **Negative**.")

# Text input
review_input = st.text_area("✍️ Write your review here:")

# Prediction trigger
if st.button("🚀 Classify Review"):
    if review_input.strip():
        result = predict_review(review_input.strip())
        st.success(f"🎯 Prediction: **{result}**")
    else:
        st.warning("⚠️ Please enter a review before classifying.")
