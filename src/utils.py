import random
import matplotlib.pyplot as plt
import torch
from wordcloud import WordCloud

def show_sample_images(dataset, num_samples=5):
    plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        image, _, caption = dataset[idx]
        image = image.permute(1, 2, 0) * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
        image = image.numpy().clip(0, 1)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(image)
        plt.title("\n".join(caption.split()[:10]) + ('...' if len(caption.split()) > 10 else ''))
        plt.axis('off')
    plt.show()

def generate_word_cloud(vocab):
    words = list(vocab.stoi.keys())[4:]  # skip special tokens
    word_freq = " ".join(words)
    wc = WordCloud(width=800, height=400, background_color='white').generate(word_freq)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title("Vocabulary Word Cloud")
    plt.show()
