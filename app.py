import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import os
# import subprocess # No longer needed for git lfs pull
from huggingface_hub import hf_hub_download
import joblib # Use joblib for potentially safer pickle loading

# --- Use Hugging Face Hub to get file paths ---
HF_REPO_ID = "2bhavyasodhi7/rnn_toxicity"
MODEL_FILENAME = "toxicity.h5"
VECTORIZER_FILENAME = "vectorizer.pkl"

@st.cache_resource(show_spinner="Downloading model files...")
def download_files_from_hf():
    """Downloads model and vectorizer, returns their local cached paths."""
    try:
        model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
        vectorizer_path = hf_hub_download(repo_id=HF_REPO_ID, filename=VECTORIZER_FILENAME)
        return model_path, vectorizer_path
    except Exception as e:
        st.error(f"❌ Failed to download files from Hugging Face Hub: {e}")
        # Optionally re-raise or return None to handle the error downstream
        raise # Reraise to stop execution if download fails

# Get the paths (downloads happen here if not cached)
try:
    downloaded_model_path, downloaded_vectorizer_path = download_files_from_hf()
    st.success(f"✅ Files ready:\n - Model: {downloaded_model_path}\n - Vectorizer: {downloaded_vectorizer_path}")
except Exception:
    st.stop() # Stop the app if downloads failed

# --- Load using the downloaded paths ---

# Load model
@st.cache_resource(show_spinner="Loading model...")
def load_model(model_file_path):
    # Check existence just in case download logic had issues (optional)
    if not os.path.exists(model_file_path):
         raise FileNotFoundError(f"Model file not found at expected download path: {model_file_path}")
    # Debug: Check file size
    st.write(f"Model file size: {os.path.getsize(model_file_path)} bytes")
    return tf.keras.models.load_model(model_file_path)

# Load vectorizer (using joblib is often preferred for scikit-learn objects)
@st.cache_resource(show_spinner="Loading vectorizer...")
def load_vectorizer(vectorizer_file_path):
    if not os.path.exists(vectorizer_file_path):
         raise FileNotFoundError(f"Vectorizer file not found at expected download path: {vectorizer_file_path}")
    # Debug: Check file size
    st.write(f"Vectorizer file size: {os.path.getsize(vectorizer_file_path)} bytes")
    # SECURITY WARNING: Only load pickle files from trusted sources.
    try:
        # Using joblib which can sometimes be safer/more efficient for sklearn objects
        vectorizer = joblib.load(vectorizer_file_path)
        # Or stick to pickle if joblib doesn't work:
        # with open(vectorizer_file_path, "rb") as f:
        #     vectorizer = pickle.load(f)
        return vectorizer
    except Exception as e:
        st.error(f"❌ Error loading vectorizer pickle file: {e}")
        # Consider the security implications if loading fails strangely.
        raise # Reraise to stop execution

# Load model and vectorizer using the downloaded paths
try:
    model = load_model(downloaded_model_path)
    vectorizer = load_vectorizer(downloaded_vectorizer_path)
except Exception as e:
    st.error(f"Failed to load model or vectorizer after download.")
    st.stop()


# --- Rest of your Streamlit App UI (no changes needed below) ---

# Define labels
labels = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate']

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
        # Ensure vectorizer expects a list/iterable
        if not isinstance(user_input, list):
            input_list = [user_input]
        else:
            input_list = user_input

        # Check if the vectorizer is callable or needs .transform
        if hasattr(vectorizer, 'transform') and callable(vectorizer.transform):
             vectorized_input = vectorizer.transform(input_list)
        else:
             # Assuming the vectorizer itself is callable like in the original code
             vectorized_input = vectorizer(input_list)

        # Ensure input is suitable for model (e.g., numpy array)
        # This might depend on how the vectorizer outputs data
        if not isinstance(vectorized_input, np.ndarray):
             # TF models often expect numpy arrays. Convert if necessary.
             # This conversion step depends heavily on the vectorizer's output format.
             # If it's a sparse matrix: vectorized_input = vectorized_input.toarray()
             # If it's already dense, maybe just np.array()
             try:
                 # Attempt conversion, adjust based on actual vectorizer output
                 if hasattr(vectorized_input, 'toarray'): # Handle sparse matrix
                     vectorized_input = vectorized_input.toarray()
                 else:
                     vectorized_input = np.array(vectorized_input)
             except Exception as e:
                 st.error(f"Could not convert vectorizer output to suitable format: {e}")
                 st.stop()

        prediction = model.predict(vectorized_input)[0]

        # Display predictions
        st.subheader("🧩 Prediction Probabilities:")
        for label, prob in zip(labels, prediction):
            st.write(f"{label}: {prob * 100:.2f}%")
            st.progress(int(prob * 100))

        # Handle potential NaN or infinite values in prediction
        if np.any(np.isnan(prediction)) or np.any(np.isinf(prediction)):
            st.warning("Prediction resulted in invalid numbers.")
        else:
            top_label_idx = np.argmax(prediction)
            # Ensure index is valid
            if 0 <= top_label_idx < len(labels):
                 top_label = labels[top_label_idx]
                 st.success(f"✅ Most likely category: {top_label}")
            else:
                 st.error("Error determining the most likely category.")


# About section
st.markdown("""
---
### ℹ️ About:
This app uses an RNN-based neural network to detect toxicity levels in text comments.
Model and vectorizer downloaded from Hugging Face Hub.
""")
