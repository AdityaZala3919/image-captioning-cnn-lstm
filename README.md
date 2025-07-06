# VarnanAI: Image Captioning with CNN-LSTM

A deep learning project that generates descriptive captions for images using a CNN-LSTM architecture. The model combines a ResNet50 encoder for feature extraction with an LSTM decoder for caption generation.

🚀 **[Try the Live Demo](https://varnan-ai-image-captioning.streamlit.app/)** 🚀

## 🌟 Features

- **CNN-LSTM Architecture**: Uses ResNet50 for image feature extraction and LSTM for caption generation
- **Streamlit Web App**: Interactive interface for uploading images and generating captions
- **BLEU Score Evaluation**: Quantitative evaluation using BLEU-2 scores
- **Visualization Tools**: Sample image display, word clouds, and training loss plots
- **Configurable Training**: Flexible configuration for model parameters and training settings

## 📁 Project Structure

```
.
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration settings and paths
│   ├── dataset.py         # Dataset handling and vocabulary building
│   ├── model.py           # CNN-LSTM model architecture
│   ├── train.py           # Training loop and validation
│   ├── evaluate.py        # Model evaluation with BLEU scores
│   ├── transforms.py      # Image preprocessing transforms
│   └── utils.py           # Utility functions and visualization
├── app.py                 # Streamlit web application
└── requirements.txt       # Project dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/image-captioning.git
cd image-captioning
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the Flickr8k dataset and organize it as follows:
```
data/
└── Flickr8k/
    ├── Images/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── captions.txt
```

The `captions.txt` file should contain image filenames and their corresponding captions.

### 3. Training

```python
from src.dataset import Flickr8kDataset
from src.model import ImageCaptioning
from src.train import train
from src.transforms import get_transform
import torch

# Create dataset
dataset = Flickr8kDataset(IMAGE_DIR, CAPTIONS_FILE, transform=get_transform())

# Initialize model
model = ImageCaptioning(EMBED_SIZE, HIDDEN_SIZE, len(dataset.vocab.itos))

# Train model
train(model, dataloader, criterion, optimizer, device, epochs=10)
```

### 4. Web Application

Launch the Streamlit app locally:
```bash
streamlit run app.py
```

Or try the **[Live Demo](https://varnan-ai-image-captioning.streamlit.app/)** without any setup!

## 🏗️ Model Architecture

The model follows an encoder-decoder architecture that combines computer vision and natural language processing:

### CNN Encoder (Image Feature Extraction)
- **Base Model**: ResNet50 pretrained on ImageNet
- **Architecture**: 
  - Input: RGB images (224×224×3)
  - Feature extraction through ResNet50 backbone (removes final FC layer)
  - Global average pooling to get 2048-dimensional feature vector
  - Linear projection to 256-dimensional embedding space
  - Batch normalization for stable training
- **Output**: 256-dimensional feature vectors representing image content
- **Frozen Weights**: CNN weights can be frozen or fine-tuned based on `train_cnn` parameter

### LSTM Decoder (Caption Generation)
- **Word Embedding**: 
  - Converts word indices to dense 256-dimensional vectors
  - Vocabulary size depends on frequency threshold (default: words appearing ≥5 times)
- **LSTM Architecture**:
  - Hidden state size: 512 dimensions
  - Single layer LSTM (configurable)
  - Hidden and cell states initialized from image features via linear projections
  - Processes word embeddings sequentially
- **Output Layer**: Linear layer mapping LSTM hidden states to vocabulary logits
- **Special Tokens**: `<SOS>` (start), `<EOS>` (end), `<PAD>` (padding), `<UNK>` (unknown)

### Training Process
1. **Image Encoding**: CNN encoder extracts visual features from input images
2. **State Initialization**: Image features initialize LSTM hidden and cell states
3. **Teacher Forcing**: During training, ground truth captions are fed as input
4. **Sequential Prediction**: LSTM predicts next word at each time step
5. **Loss Calculation**: Cross-entropy loss between predictions and target sequences

### Inference Process
1. **Feature Extraction**: Image passed through CNN encoder
2. **Sequence Generation**: Starting with `<SOS>` token, LSTM generates words autoregressively
3. **Stopping Criterion**: Generation stops at `<EOS>` token or maximum length (20 words)
4. **Caption Assembly**: Generated word indices converted back to text using vocabulary

### Key Design Decisions
- **Attention Mechanism**: Not implemented - uses global image features only
- **Beam Search**: Not implemented - uses greedy decoding
- **Caption Length**: Maximum 20 words to balance quality and computational efficiency
- **Vocabulary Pruning**: Words appearing less than 5 times treated as `<UNK>`

## 📊 Configuration

Key parameters in `src/config.py`:

```python
EMBED_SIZE = 256        # Embedding dimension
HIDDEN_SIZE = 512       # LSTM hidden state size
NUM_LAYERS = 1          # Number of LSTM layers
BATCH_SIZE = 32         # Training batch size
LEARNING_RATE = 1e-3    # Learning rate
FREQ_THRESHOLD = 5      # Minimum word frequency for vocabulary
```

## 🎯 Training Process

1. **Data Preprocessing**: Images are resized to 224x224 and normalized using ImageNet statistics
2. **Vocabulary Building**: Creates vocabulary from captions with frequency threshold
3. **Model Training**: Uses teacher forcing with cross-entropy loss
4. **Validation**: Computes validation loss and BLEU scores
5. **Visualization**: Plots training curves and sample predictions

## 📈 Evaluation

The model is evaluated using:
- **BLEU-2 Score**: Measures similarity between generated and reference captions
- **Visual Inspection**: Displays sample images with generated captions
- **Loss Tracking**: Monitors training and validation loss curves

## 🔧 Usage Examples

### Generate Caption for Single Image
```python
from src.model import ImageCaptioning
from PIL import Image
import torch

# Load model and generate caption
model = ImageCaptioning(EMBED_SIZE, HIDDEN_SIZE, vocab_size)
model.load_state_dict(torch.load(MODEL_PATH))

image = Image.open('path/to/image.jpg')
caption = model.generate_caption(image_tensor, vocab)
print(f"Generated caption: {caption}")
```

### Evaluate Model Performance
```python
from src.evaluate import evaluate

# Evaluate on test set
evaluate(model, test_dataloader, vocab, device)
```

## 🎨 Visualization Features

- **Sample Images**: Display random dataset samples with captions
- **Word Clouds**: Visualize vocabulary distribution
- **Training Curves**: Plot loss progression over epochs
- **Prediction Examples**: Show model predictions alongside ground truth

## 🛠️ Dependencies

- `torch>=1.13.0` - Deep learning framework
- `torchvision>=0.14.0` - Computer vision utilities
- `streamlit` - Web application framework
- `pandas` - Data manipulation
- `nltk` - Natural language processing
- `matplotlib` - Plotting and visualization
- `Pillow` - Image processing
- `wordcloud` - Word cloud generation

## 📝 Model Details

- **Input**: RGB images of any size (resized to 224x224)
- **Output**: Natural language captions
- **Training Data**: Flickr8k dataset with ~8,000 images and 40,000 captions
- **Vocabulary Size**: Configurable based on word frequency threshold
- **Maximum Caption Length**: 20 words (configurable)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ResNet50 architecture from torchvision
- Flickr8k dataset creators
- NLTK library for text processing
- Streamlit for the web interface

## 📧 Contact

For questions or suggestions, please open an issue or contact [adityazala404@gmail.com](adityazala404@gmail.com).
