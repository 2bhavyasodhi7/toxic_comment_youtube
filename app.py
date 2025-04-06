import streamlit as st
import requests
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tempfile
import os

# Hugging Face model URL (.h5 file only)
HUGGINGFACE_MODEL_URL = "https://huggingface.co/spaces/2bhavyasodhi7/transformer_comment_toxicity/resolve/main/tf_model.h5"

# Constants
MAX_LEN = 200  # Must match training config
VOCAB_SIZE = 10000  # Must match training config

@st.cache_resource
def load_model_from_huggingface():
    model_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
    model_file.write(requests.get(HUGGINGFACE_MODEL_URL).content)
    model_file.close()
    return load_model(model_file.name)

@st.cache_resource
def create_default_tokenizer():
    # WARNING: This will NOT produce accurate predictions unless your model was trained with same tokenizer
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    # Placeholder fit on dummy data just to make it usable — you MUST use the original tokenizer for real accuracy
    tokenizer.fit_on_texts(["This is a placeholder fit."])
    return tokenizer

# Load model and tokenizer
model = load_model_from_huggingface()
tokenizer = create_default_tokenizer()

# Streamlit UI
st.set_page_config(page_title="Toxic Comment Checker", layout="centered")
st.title("🧪 Toxic Comment Detection")
st.write("Enter a comment below and check if it's toxic.")

user_input = st.text_area("Type your comment here...")

if st.button("Check Toxicity"):
    if user_input.strip() == "":
        st.warning("Please enter a comment first.")
    else:
        # Preprocess input
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding="post", truncating="post")

        # Predict
        prediction = model.predict(padded)[0][0]

        # Show result
        st.subheader("Prediction")
        st.write(f"Toxicity Score: `{prediction:.4f}`")
        if prediction > 0.5:
            st.error("⚠️ This comment is likely **toxic**.")
        else:
            st.success("✅ This comment is **not toxic**.")
