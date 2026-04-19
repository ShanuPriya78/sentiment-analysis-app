import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------- DATA ----------
data = {
    'text': [
        "I love this product",
        "This is amazing",
        "Very bad experience",
        "I hate this",
        "Not good at all",
        "Absolutely fantastic",
        "Worst purchase ever",
        "Really happy with this"
    ],
    'label': [1, 1, 0, 0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# ---------- CLEANING ----------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

# ---------- MODEL ----------
X = df['text']
y = df['label']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# ---------- UI ----------
st.title("💬 Sentiment Analysis App")

user_input = st.text_input("Enter a sentence:")

if user_input:
    clean_input = clean_text(user_input)
    vector_input = vectorizer.transform([clean_input])
    prediction = model.predict(vector_input)
    prob = model.predict_proba(vector_input)

    if prediction[0] == 1:
        st.success("Positive 😊")
    else:
        st.error("Negative 😡")
    st.write(f"Confidence: {prob.max()*100:.2f}%")    