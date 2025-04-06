import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os

# 🛠 Ensure Git LFS files are pulled (important for Streamlit Cloud!)
os.system('git lfs pull')

# Paths to local files
MODEL_PATH = "toxicity.h5"
VECTORIZER_PATH = "vectorizer.pkl"

# Cache loading of model and vectorizer
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_resource(show_spinner="Loading vectorizer...")
def load_vectorizer():
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
