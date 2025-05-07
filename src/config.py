import os
import torch

# Automatically determine the ROOT_DIR path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data paths
IMAGE_DIR = os.path.join(ROOT_DIR, "data", "Flickr8k", "Images")
CAPTIONS_FILE = os.path.join(ROOT_DIR, "data", "Flickr8k", "captions.txt")

# Model save/load paths
MODEL_PATH = os.path.join(ROOT_DIR, "models", "cnn_lstm_captioning.pth")
VOCAB_PATH = os.path.join(ROOT_DIR, "models", "vocab.pkl")

# Training settings
EMBED_SIZE = 256
HIDDEN_SIZE = 512
NUM_LAYERS = 1
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
FREQ_THRESHOLD = 5

# Device config
DEVICE = "cuda" if os.environ.get("USE_CUDA", "0") == "1" and torch.cuda.is_available() else "cpu"
