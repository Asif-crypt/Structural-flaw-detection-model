import argparse
import os

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(32, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(32, 16)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        bottleneck = self.bottleneck(self.pool2(d2))
        u2 = self.conv_up2(torch.cat((d2, self.up2(bottleneck)), dim=1))
        u1 = self.conv_up1(torch.cat((d1, self.up1(u2)), dim=1))
        return self.out_conv(u1)


class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=(256, 256), max_samples=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        image_names = [name for name in os.listdir(img_dir) if name.lower().endswith((".jpg", ".jpeg", ".png"))]
        self.img_names = image_names[:max_samples] if max_samples else image_names

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image_name = self.img_names[idx]
        base_name, _ = os.path.splitext(image_name)
        mask_name = base_name + ".png"

        image_path = os.path.join(self.img_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found or unreadable: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0
        return image_tensor, mask_tensor


def dice_score(logits, masks, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return ((2 * intersection + eps) / (union + eps)).mean().item()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Tiny U-Net crack segmenter.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training U-Net on {device}")

    base_dir = os.path.join("data", "segmentation", "deepcrack")
    train_dataset = CrackDataset(
        os.path.join(base_dir, "train_img"),
        os.path.join(base_dir, "train_lab"),
        size=(args.size, args.size),
        max_samples=args.max_samples,
    )
    val_dataset = CrackDataset(
        os.path.join(base_dir, "test_img"),
        os.path.join(base_dir, "test_lab"),
        size=(args.size, args.size),
        max_samples=args.max_samples,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = TinyUNet().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_dice = -1.0
    os.makedirs(os.path.join("models", "unet"), exist_ok=True)
    save_path = os.path.join("models", "unet", "best_unet.pth")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                val_loss += criterion(logits, masks).item() * images.size(0)
                val_dice += dice_score(logits, masks) * images.size(0)

        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        val_dice /= len(val_dataset)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_dice: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print(f"U-Net training complete. Best validation Dice: {best_dice:.4f}")
