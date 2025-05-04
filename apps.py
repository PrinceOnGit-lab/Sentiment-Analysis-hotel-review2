import streamlit as st
import pickle
import re
from textblob import TextBlob

# Load the trained model and vectorizer
with open("svc_model1.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# App UI
st.title("🏨 Hotel Review Sentiment Analysis")
st.write("Enter a hotel review to predict if it's **Positive**, **Neutral**, or **Negative**.")
st.caption("Note: Avoid using numbers, ratings like '5*', placeholder text, or spelling mistakes.")

user_review = st.text_area("✏️ Enter your review:")

# Detects placeholder text
def is_placeholder_input(text):
    text = text.strip().lower()
    placeholder_phrases = [
        "write something about hotel",
        "say something about hotel",
        "type a review",
        "write a review",
        "enter your review",
        "say something",
        "test review"
    ]
    return any(phrase in text for phrase in placeholder_phrases)

# Checks for digits, star ratings, and gibberish
def is_valid_review(text):
    if re.search(r'\d|\d\*', text):  # Contains digits or '5*'
        return False
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return len(words) >= 3

# Checks for spelling mistakes
def has_too_many_spelling_errors(text, threshold=0.4):
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    if not words:
        return True
    misspelled = 0
    for word in words:
        blob = TextBlob(word)
        if word != blob.correct().string:
            misspelled += 1
    error_ratio = misspelled / len(words)
    return error_ratio >= threshold

# Button handler
if st.button("🔍 Predict Sentiment"):
    if user_review.strip():
        if is_placeholder_input(user_review):
            st.error("❌ This input looks like a placeholder. Please write a real review about the hotel.")
        elif not is_valid_review(user_review):
            st.error("❌ Invalid input: Avoid digits, star ratings like '5*', or gibberish.")
        elif has_too_many_spelling_errors(user_review):
            st.error("❌ Please check your spelling.")
        else:
            # Process and predict
            processed_review = user_review.lower()
            review_vectorized = vectorizer.transform([processed_review])
            prediction = model.predict(review_vectorized)[0]
            sentiment = (
                "Negative" if prediction == 0 else
                "Positive" if prediction == 1 else
                "Neutral"
            )
            st.success(f"✅ **Predicted Sentiment:** {sentiment}")
    else:
        st.warning("⚠️ Please enter a review before clicking Predict.")
