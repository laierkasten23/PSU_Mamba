import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SegmentationDataset3D
from model import ViT3DSegmenter
from utils import dice_coefficient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 2
epochs = 100
lr = 1e-4
json_path = "config.json"

# Setup
dataset = SegmentationDataset3D(json_path)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = ViT3DSegmenter().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device).unsqueeze(1).float()
        preds = model(x)
        loss = criterion(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / len(train_loader):.4f}")
    torch.save(model.state_dict(), f"model_epoch{epoch+1}.pth")
