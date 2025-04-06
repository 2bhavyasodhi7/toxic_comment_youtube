import streamlit as st
import os
import numpy as np
import tensorflow as tf
import pickle

# Force Git LFS pull
if not os.path.exists("toxicity.h5") or os.path.getsize("toxicity.h5") < 1000000:
    st.warning("Running Git LFS pull to fetch model files...")
    os.system("git lfs install")
    os.system("git lfs pull")
    st.success("Git LFS files pulled!")

# Check if model file exists and is valid size
if not os.path.exists("toxicity.h5"):
    st.error("toxicity.h5 file not found after git lfs pull!")
    st.stop()

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("toxicity.h5")

# Load vectorizer
@st.cache_resource
def load_vectorizer():
    with open("vectorizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
vectorizer = load_vectorizer()

labels = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']

st.title("📝 Toxic Comment Classification App")

user_input = st.text_area("Enter your comment:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        vectorized_input = vectorizer([user_input])
        prediction = model.predict(vectorized_input)[0]

        st.subheader("Prediction Probabilities:")
        for label, prob in zip(labels, prediction):
            st.write(f"{label}: {prob*100:.2f}%")
            st.progress(int(prob*100))
        st.success(f"Most likely: {labels[np.argmax(prediction)]}")
