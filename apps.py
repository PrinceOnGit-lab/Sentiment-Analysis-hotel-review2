import streamlit as st
import pickle
import re

# Load the trained model and vectorizer
with open("svc_model1.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Streamlit App
st.title("Hotel Review Sentiment Analysis")
st.write("Enter a hotel review to predict if it's Positive, Neutral, or Negative.")

# User input
user_review = st.text_area("Enter your review:")

def is_valid_review(text):
    """
    Checks if the input:
    - Doesn't contain digits
    - Doesn't contain star rating patterns like '5*' or '4*'
    - Has at least 3 valid words
    """
    if re.search(r'\d', text):  # Contains any digit
        return False
    if re.search(r'\d\*', text):  # Contains '5*', '4*', etc.
        return False
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    return len(words) >= 3

if st.button("Predict Sentiment"):
    if user_review.strip():
        if not is_valid_review(user_review):
            st.error("Invalid input: Please enter a valid review without numbers or special characters."
                     "And Please avoid numbers, ratings like '5*', or gibberish.")
        else:
            # Preprocessing
            processed_review = user_review.lower()

            # Vectorize
            review_vectorized = vectorizer.transform([processed_review])

            # Predict
            prediction = model.predict(review_vectorized)[0]
            sentiment = (
                "Negative" if prediction == 0 else
                "Positive" if prediction == 1 else
                "Neutral"
            )

            # Output
            st.success(f"**Predicted Sentiment:** {sentiment}")
    else:
        st.warning("Please enter a review before clicking Predict.")
