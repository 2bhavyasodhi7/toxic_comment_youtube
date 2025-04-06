import streamlit as st
import requests
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

HUGGINGFACE_MODEL_URL = "https://huggingface.co/2bhavyasodhi7/transformer_toxicity/resolve/main/tf_model.h5"
MODEL_PATH = "tf_model.h5"

# Dummy tokenizer for demo purposes
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(["this is toxic", "this is clean", "great", "stupid"])  # just for structure

MAX_LEN = 100

@st.cache_resource
def load_model_once():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model... Please wait."):
            response = requests.get(HUGGINGFACE_MODEL_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    return load_model(MODEL_PATH)

model = load_model_once()

st.title("🧪 Toxic Comment Detection")
user_input = st.text_area("✍️ Type your comment here...")

if st.button("Check for Toxicity"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        sequence = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
        prediction = model.predict(padded)[0][0]

        st.write("### 🧾 Result:")
        if prediction > 0.5:
            st.error(f"🚨 Toxic Comment Detected! (Score: {prediction:.2f})")
        else:
            st.success(f"✅ Comment seems safe. (Score: {prediction:.2f})")
