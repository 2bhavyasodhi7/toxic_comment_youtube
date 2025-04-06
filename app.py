import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
import subprocess

# Paths to model and vectorizer
MODEL_PATH = "toxicity.h5"
VECTORIZER_PATH = "vectorizer.pkl"

# Show files in current directory (debugging)
st.write("📁 Files in current directory:", os.listdir())

# Run Git LFS pull if model/vectorizer is missing
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    st.warning("Model or vectorizer file not found. Running Git LFS pull...")
    try:
        subprocess.run(["git", "lfs", "install"], check=True)
        subprocess.run(["git", "lfs", "pull"], check=True)
        st.success("✅ Git LFS files pulled!")
        st.write("📁 Files after pull:", os.listdir())
    except Exception as e:
        st.error(f"❌ Git LFS pull failed: {e}")

# Load model
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found!")
    return tf.keras.models.load_model(MODEL_PATH)

# Load vectorizer
@st.cache_resource(show_spinner="Loading vectorizer...")
def load_vectorizer():
    if not os.path.exists(VECTORIZER_PATH):
        raise FileNotFoundError(f"{VECTORIZER_PATH} not found!")
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer

# Load model and vectorizer
model = load_model()
vectorizer = load_vectorizer()

# Define labels
labels = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']

# Streamlit App UI
st.title("📝 Toxic Comment Classification App")
st.markdown("""
Type a comment below, and the model will predict the probability of it being:
- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate
""")

# Text input
user_input = st.text_area("Enter your comment:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text for prediction.")
    else:
        # Preprocess and predict
        vectorized_input = vectorizer([user_input])
        prediction = model.predict(vectorized_input)[0]

        # Display predictions
        st.subheader("🧩 Prediction Probabilities:")
        for label, prob in zip(labels, prediction):
            st.write(f"{label}: {prob * 100:.2f}%")
            st.progress(int(prob * 100))

        top_label = labels[np.argmax(prediction)]
        st.success(f"✅ Most likely category: {top_label}")

# About section
st.markdown("""
---
### ℹ️ About:
This app uses an RNN-based neural network to detect toxicity levels in text comments.
""")
