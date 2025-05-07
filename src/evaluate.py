import numpy as np
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu
import matplotlib.pyplot as plt

def evaluate(model, dataloader, vocab, device):
    model.eval()
    bleu_scores = []
    shown = 0

    with torch.no_grad():
        for images, _, _, raw_captions in dataloader:
            for i in range(len(images)):
                image = images[i].to(device)
                generated = model.generate_caption(image, vocab)
                reference = [nltk.word_tokenize(raw_captions[i].lower())]
                candidate = nltk.word_tokenize(generated.lower())
                bleu = sentence_bleu(reference, candidate, weights=(0.5, 0.5)) #bleu2
                bleu_scores.append(bleu)

                if shown < 5:
                    img = image.cpu().permute(1, 2, 0) * torch.tensor([0.229, 0.224, 0.225]) + \
                          torch.tensor([0.485, 0.456, 0.406])
                    img = img.numpy().clip(0, 1)
                    plt.figure(figsize=(6, 4))
                    plt.imshow(img)
                    plt.title(f"GT: {raw_captions[i][:60]}...\nPred: {generated[:60]}...", fontsize=9)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.show()
                    shown += 1
            if shown >= 5:
                break

    print(f"Average BLEU score: {np.mean(bleu_scores):.4f}")
