import torch
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

def train(model, dataloader, criterion, optimizer, device, epochs=10, val_loader=None):
    model.train()
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        total_loss = 0
        for images, captions, lengths, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, captions = images.to(device), captions.to(device)
            optimizer.zero_grad()
            outputs = model(images, captions[:, :-1], [l - 1 for l in lengths])
            targets = pack_padded_sequence(captions[:, 1:], [l - 1 for l in lengths], batch_first=True, enforce_sorted=False)[0]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(dataloader)
        train_losses.append(avg_train_loss)

        if val_loader:
            val_loss = compute_val_loss(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

    plt.plot(train_losses, label="Train Loss")
    if val_loader:
        plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

def compute_val_loss(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, captions, lengths, _ in dataloader:
            images, captions = images.to(device), captions.to(device)
            outputs = model(images, captions[:, :-1], [l - 1 for l in lengths])
            targets = pack_padded_sequence(captions[:, 1:], [l - 1 for l in lengths], batch_first=True, enforce_sorted=False)[0]
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    model.train()
    return val_loss / len(dataloader)
