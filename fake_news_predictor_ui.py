import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ✅ Load model and vectorizer from Google Drive CSV
@st.cache_resource
def load_model():
    # Link to your Google Drive CSV file
    drive_url = "https://drive.google.com/uc?id=17wCEtOx4DgJtdfNzfM9F9RNbzzaTh4b0"

    # Load the CSV from Google Drive
    df = pd.read_csv(drive_url)

    # Convert labels to 0 and 1
    df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

    # Features and target
    X = df['text']
    y = df['label']

    # Vectorize the text
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_vec = vectorizer.fit_transform(X)

    # Train the model
    model = LogisticRegression()
    model.fit(X_vec, y)

    return model, vectorizer

# Load model and vectorizer
model, vectorizer = load_model()

# 🎨 UI Configuration
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.markdown("Paste a news article or headline below and check whether it's **REAL** or **FAKE** using machine learning.")

# ✍️ Input Field
user_input = st.text_area("Enter the news content:", height=200, placeholder="e.g., Government announces new policy...")

# 🔍 Predict Button
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        prob = model.predict_proba(input_vec)[0][prediction] * 100
        result = "✅ REAL NEWS" if prediction == 1 else "🚨 FAKE NEWS"
        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {prob:.2f}%")
