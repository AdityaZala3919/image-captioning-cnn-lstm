import streamlit as st
import torch
from PIL import Image
import pickle
import os
import matplotlib.pyplot as plt

from src.transforms import get_transform
from src.model import ImageCaptioning
from src.dataset import Vocabulary
from src.config import MODEL_PATH, VOCAB_PATH, EMBED_SIZE, HIDDEN_SIZE, DEVICE

# --- Streamlit page config ---
st.set_page_config(page_title="Image Captioning", layout="centered")
st.title("🖼️ Image Captioning App")
st.write("Upload an image, and the model will generate a caption for it.")

# --- Load model and vocab ---
@st.cache_resource
def load_model():
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)

    model = ImageCaptioning(EMBED_SIZE, HIDDEN_SIZE, len(vocab.itos)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model, vocab

model, vocab = load_model()

# --- Image transformation ---
transform = get_transform()

# --- Upload image ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            image_tensor = transform(image).to(DEVICE)
            caption = model.generate_caption(image_tensor, vocab)
            st.success("Caption Generated!")
            st.markdown(f"**Caption**: {caption}")
