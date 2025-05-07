import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision.models import resnet50, ResNet50_Weights


class CNNEncoder(nn.Module):
    def __init__(self, embed_size, train_cnn=False):
        super(CNNEncoder, self).__init__()
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size)

        for param in self.resnet.parameters():
            param.requires_grad = train_cnn

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.fc(features))
        return features

class LSTMDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(LSTMDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        h0 = self.init_h(features).unsqueeze(0)
        c0 = self.init_c(features).unsqueeze(0)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        hiddens, _ = self.lstm(packed, (h0, c0))
        outputs = self.linear(hiddens[0])
        return outputs

class ImageCaptioning(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, train_cnn=False):
        super(ImageCaptioning, self).__init__()
        self.encoder = CNNEncoder(embed_size, train_cnn)
        self.decoder = LSTMDecoder(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions, lengths):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        return outputs

    def generate_caption(self, image, vocab, max_len=20):
        with torch.no_grad():
            feature = self.encoder(image.unsqueeze(0))
            h = self.decoder.init_h(feature).unsqueeze(0)
            c = self.decoder.init_c(feature).unsqueeze(0)
            inputs = torch.tensor([vocab.stoi["<SOS>"]], dtype=torch.long).unsqueeze(0)
            caption = []

            for _ in range(max_len):
                embeddings = self.decoder.embed(inputs)
                hiddens, (h, c) = self.decoder.lstm(embeddings, (h, c))
                output = self.decoder.linear(hiddens.squeeze(1))
                predicted = output.argmax(1)
                word = vocab.itos[predicted.item()]
                if word == "<EOS>":
                    break
                caption.append(word)
                inputs = predicted.unsqueeze(1)
                
        return ' '.join(caption)
