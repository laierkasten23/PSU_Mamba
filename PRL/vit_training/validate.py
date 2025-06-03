import torch
from torch.utils.data import DataLoader
from dataset import SegmentationDataset3D
from model import ViT3DSegmenter
from utils import dice_coefficient

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViT3DSegmenter().to(device)
model.load_state_dict(torch.load("model_epoch100.pth"))
model.eval()

val_dataset = SegmentationDataset3D("config.json")
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

total_dice = 0
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(device), y.to(device).unsqueeze(1).float()
        preds = model(x)
        dice = dice_coefficient(preds, y)
        total_dice += dice.item()

print(f"Average Dice: {total_dice / len(val_loader):.4f}")
