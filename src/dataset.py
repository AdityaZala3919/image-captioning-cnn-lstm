import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import nltk
from torch.nn.utils.rnn import pad_sequence

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def build_vocab(self, captions):
        nltk.download('punkt')
        from collections import Counter
        counter = Counter()
        for caption in captions:
            tokens = nltk.word_tokenize(caption.lower())
            counter.update(tokens)
        idx = 4
        for word, freq in counter.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokens = nltk.word_tokenize(text.lower())
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]

class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None, vocab=None):
        self.df = pd.read_csv(captions_file)
        self.image_dir = image_dir
        self.transform = transform
        self.df['image'] = self.df['image'].apply(lambda x: os.path.join(image_dir, x))
        self.captions = self.df['caption'].tolist()
        self.images = self.df['image'].tolist()

        if vocab is None:
            self.vocab = Vocabulary()
            self.vocab.build_vocab(self.captions)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        numericalized = [self.vocab.stoi["<SOS>"]] + \
                        self.vocab.numericalize(caption) + \
                        [self.vocab.stoi["<EOS>"]]
        
        return image, torch.tensor(numericalized), caption

def collate_fn(batch):
    images, captions, raw_captions = zip(*batch)
    images = torch.stack(images)
    lengths = [len(c) for c in captions]
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions, lengths, raw_captions
