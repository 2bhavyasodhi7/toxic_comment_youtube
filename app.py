import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
import requests

# URLs to Hugging Face
MODEL_URL = "https://huggingface.co/2bhavyasodhi7/rnn_toxicity/resolve/main/toxicity.h5"
VECTORIZER_URL = "https://huggingface.co/2bhavyasodhi7/rnn_toxicity/resolve/main/vectorizer.pkl"

MODEL_PATH = "toxicity.h5"
VECTORIZER_PATH = "vectorizer.pkl"

# Download function with caching
@st.cache_resource(show_spinner=False)
def download_file(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename}..."):
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, stream=True)

            if response.status_code == 200:
                with open(filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                st.success(f"{filename} downloaded successfully!")
            else:
                st.error(f"Failed to download {filename}. Status code: {response.status_code}")
    return filename

# Cached loading of model and vectorizer
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model_file = download_file(MODEL_URL, MODEL_PATH)
    model = tf.keras.models.load_model(model_file)
    return model

@st.cache_resource(show_spinner="Loading vectorizer...")
def load_vectorizer():
    vectorizer_file = download_file(VECTORIZER_URL, VECTORIZER_PATH)
    with open(vectorizer_file, "rb") as f:
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
        st.warning("Please enter some text for prediction.")
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
        st.success(f"🧩 Most likely category: {top_label}")

# About section
st.markdown("""
---
### ℹ About:
This app uses an RNN-based neural network to detect toxicity levels in text comments.
""")
