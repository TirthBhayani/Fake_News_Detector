import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

@st.cache_resource
def load_model():
    file_path = "fake_or_real_news.csv"  # File should be in the same directory

    try:
        df = pd.read_csv(file_path, encoding="utf-8", on_bad_lines='skip')
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()

    # Clean column headers
    df.columns = df.columns.str.strip().str.lower()
    st.write("✅ Columns loaded:", df.columns.tolist())  # Debug step

    # Check required columns
    if 'text' not in df.columns or 'label' not in df.columns:
        st.error("❌ 'text' and 'label' columns not found. Please check your dataset.")
        st.stop()

    # Encode labels
    df['label'] = df['label'].map({'fake': 0, 'real': 1})

    X = df['text']
    y = df['label']

    # Vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_vec = vectorizer.fit_transform(X)

    # Model Training
    model = LogisticRegression()
    model.fit(X_vec, y)

    return model, vectorizer

model, vectorizer = load_model()

st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detector")
st.write("Detect whether a news article is **REAL** or **FAKE** using machine learning.")

user_input = st.text_area("✍️ Paste the news content or headline below:", height=200)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)[0]
        prob = model.predict_proba(input_vec)[0][prediction] * 100
        result = "✅ REAL NEWS" if prediction == 1 else "🚨 FAKE NEWS"
        st.success(f"Prediction: {result} ({prob:.2f}% confidence)")
