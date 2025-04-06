import streamlit as st
import requests
import numpy as np
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Replace this with the correct model URL from your Hugging Face *Model Repository*
HUGGINGFACE_MODEL_URL = "https://huggingface.co/your-username/your-model-repo/resolve/main/tf_model.h5"

# A simple tokenizer as a placeholder
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
# Fit on dummy data to avoid errors — in real cases, this must be the **same tokenizer** used during training
dummy_texts = ["example", "toxic", "clean", "not toxic"]
tokenizer.fit_on_texts(dummy_texts)

MAX_LEN = 100  # Replace with actual max length used during model training

@st.cache_resource
def load_model_from_huggingface():
    model_file = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")

    response = requests.get(HUGGINGFACE_MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code != 200:
        raise ValueError("Failed to download model file. Check the URL.")

    model_file.write(response.content)
    model_file.close()

    return load_model(model_file.name)

# Load model
model = load_model_from_huggingface()

# Streamlit App UI
st.set_page_config(page_title="Toxic Comment Checker", layout="centered")
st.title("🧪 Toxic Comment Detection")
st.write("Enter a comment below and check if it's toxic.")

user_input = st.text_area("Type your comment here...")

if st.button("Check for Toxicity"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        # Tokenize and pad input
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

        # Predict
        prediction = model.predict(padded)[0][0]  # Adjust indexing if needed

        st.write("### Result:")
        if prediction > 0.5:
            st.error(f"🚨 Toxic Comment Detected! (Score: {prediction:.2f})")
        else:
            st.success(f"✅ Comment seems safe. (Score: {prediction:.2f})")
