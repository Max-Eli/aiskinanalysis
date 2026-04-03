"""
Training script for SkinAnalysisNet (EfficientNet-B0).

Usage:
  python train.py --data_dir /path/to/dataset --epochs 30 --batch_size 32

Dataset format expected:
  data_dir/
    images/
      <id>.jpg
    labels.csv   # columns: image_id, acne, hyperpigmentation, melasma, redness
                 # values: float 0.0–1.0

Outputs:
  cv_service/weights/skin_model.pt
"""

import argparse
import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from models.efficientnet import SkinAnalysisNet, CONCERNS


class SkinDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.img_dir = os.path.join(data_dir, "images")
        self.transform = transform
        self.records = []
        with open(os.path.join(data_dir, "labels.csv")) as f:
            for row in csv.DictReader(f):
                self.records.append({
                    "image_id": row["image_id"],
                    "labels": {c: float(row[c]) for c in CONCERNS},
                })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(os.path.join(self.img_dir, rec["image_id"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = torch.tensor([rec["labels"][c] for c in CONCERNS], dtype=torch.float32)
        return img, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    full = SkinDataset(args.data_dir, transform=train_tf)
    val_size = max(1, int(0.15 * len(full)))
    train_ds, val_ds = random_split(full, [len(full) - val_size, val_size])
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    model = SkinAnalysisNet(pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    os.makedirs("weights", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            pred_tensor = torch.stack([preds[c] for c in CONCERNS], dim=1)
            loss = criterion(pred_tensor, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs)
                pred_tensor = torch.stack([preds[c] for c in CONCERNS], dim=1)
                val_loss += criterion(pred_tensor, labels).item()

        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{args.epochs} — train: {train_loss/len(train_loader):.4f}  val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "weights/skin_model.pt")
            print("  ✓ Saved best model.")

    print("Training complete. Weights at: cv_service/weights/skin_model.pt")


if __name__ == "__main__":
    main()
