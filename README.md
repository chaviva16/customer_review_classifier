## 🍽️ Restaurant Review Sentiment Classifier

This Streamlit app allows users to classify restaurant reviews as **Positive 😊** or **Negative 😞** using a machine learning model trained on 10,000+ labeled reviews.

🔗 [Live App](https://customer-review-classifier.streamlit.app)

---

## 🚀 Features

- 📝 Enter any restaurant review and get instant sentiment prediction
- 🧠 Model: Naive Bayes with TF-IDF vectorization
- ⚡ Fast, lightweight, and easy to use
- ✅ Clean UI

---

## 🧠 How It Works

1. Review text is preprocessed (cleaned, tokenized, stemmed)
2. Text is transformed using **TF-IDF Vectorizer**
3. A **Naive Bayes classifier** predicts whether the review is positive or negative
4. Results are displayed in real time on the app

---

## 📁 Files Included

| File                      | Description                                      |
|---------------------------|--------------------------------------------------|
| `restaurant_review_app.py`| Streamlit app source code                        |
| `naive_bayes_model.pkl`   | Trained Naive Bayes model (saved with joblib)    |
| `tfidf_vectorizer.pkl`    | Fitted TF-IDF vectorizer                         |
| `requirements.txt`        | Dependencies for running the app                 |

---

## 🧪 Example Inputs

| Review                                             | Prediction |
|----------------------------------------------------|------------ |
| "The food was amazing and the service was great!"  |🟢 Positive  |
| "Terrible service, the food was cold and bland."   | 🔴 Negative |

---

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
Improve model accuracy,
Optimize preprocessing steps,
Enhance the Streamlit UI.

## License 📜
This project is licensed under the MIT License.

## Contact 💌
For questions or suggestions, feel free to open an issue or connect via GitHub.

