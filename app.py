import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
import requests

# URLs to Hugging Face
MODEL_URL = "https://huggingface.co/2bhavyasodhi7/rnn_toxicity/resolve/main/toxic_comment_rnn.h5"
VECTORIZER_URL = "https://huggingface.co/2bhavyasodhi7/rnn_toxicity/resolve/main/vectorizer.pkl"

MODEL_PATH = "toxic_comment_rnn.h5"
VECTORIZER_PATH = "vectorizer.pkl"

# Download function
def download_file(url, filename):
    if not os.path.exists(filename):
        with st.spinner(f"Downloading {filename}..."):
            response = requests.get(url, stream=True)
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        st.success(f"{filename} downloaded successfully!")

# Download model and vectorizer
download_file(MODEL_URL, MODEL_PATH)
download_file(VECTORIZER_URL, VECTORIZER_PATH)

# Load the model and vectorizer
model = tf.keras.models.load_model(MODEL_PATH)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

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
            st.write(f"{label}:** {prob * 100:.2f}%")
            st.progress(int(prob * 100))
            
        top_label = labels[np.argmax(prediction)]
        st.success(f"🧩 Most likely category: *{top_label}*")

# About section
st.markdown("""
---
### ℹ About:
This app uses an RNN-based neural network to detect toxicity levels in text comments.
""")
