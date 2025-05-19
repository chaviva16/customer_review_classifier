## Customer Review Classifier 📝✨


## Overview
This project is a Machine Learning-based Customer Review Classifier built using Python, NLTK, Scikit-learn, and Streamlit.
It analyzes customer reviews and predicts whether they are Positive or Negative using Natural Language Processing (NLP).

## Features 🚀
✅ Preprocesses customer reviews (tokenization, stopword removal, stemming)

✅ SVM model trained on CountVectorized data

✅ Interactive Streamlit web app for user-friendly review classification

✅ Fast and efficient machine learning text classifier

## Technologies Used 🛠
Python – Data processing

NLTK – Text preprocessing (stopwords, stemming, tokenization)

Scikit-learn – Machine learning algorithms

Streamlit – Web interface

Joblib – Model and vectorizer serialization

## Installation 💻
Step 1: Clone the Repository
bash
git clone https://github.com/chaviva16/customer_review_classifier
cd customer_review_classifier

Step 2: Install Dependencies
bash
pip install -r requirements.txt
Make sure your requirements.txt includes:
streamlit
nltk
scikit-learn
joblib

Step 3: Download NLTK Stopwords
Open Python and run:
import nltk
nltk.download("stopwords")

Step 4: Run the Streamlit App
bash
Copy
Edit
streamlit run customer_review.py


## Usage 🏆
Launch the Streamlit app.
Enter a customer review in the text box.

Click “Classify Review”.

View the prediction: Positive or Negative.

## Contributing 🤝
You’re welcome to contribute!

You can:
Improve model accuracy
Optimize preprocessing steps
Enhance the Streamlit UI

## License 📜
This project is licensed under the MIT License.

## Contact 💌
For questions or suggestions, feel free to open an issue or connect via GitHub.

