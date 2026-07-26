import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(TinyUNet, self).__init__()
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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        b = self.bottleneck(p2)
        u2 = self.up2(b)
        u2_conv = self.conv_up2(torch.cat((d2, u2), dim=1))
        u1 = self.up1(u2_conv)
        u1_conv = self.conv_up1(torch.cat((d1, u1), dim=1))
        return self.sigmoid(self.out_conv(u1_conv))


def run_unet_segmentation(image_path, model_path=None):
    """Runs the trained U-Net model on an image and returns the binary crack mask."""
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "unet", "best_unet.pth")

    model = TinyUNet()
    model.load_state_dict(torch.load(model_path, map_location=_DEVICE, weights_only=True))
    model.to(_DEVICE)
    model.eval()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    image_resized = cv2.resize(image, (128, 128))

    image_tensor = torch.tensor(image_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(_DEVICE)

    with torch.no_grad():
        output = model(image_tensor)

    mask = output.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    # Resize back to original
    mask = cv2.resize(mask, (orig_w, orig_h))
    return mask


def extract_crack_features(mask, image_path=None):
    """
    Extracts crack metrics from a binary segmentation mask.
    Returns a dictionary of features.
    """
    h, w = mask.shape
    total_pixels = h * w

    # Threshold to binary
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Find contours of individual crack regions
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return {
            "crack_area_px": 0,
            "crack_density": 0.0,
            "crack_count": 0,
            "mean_crack_width_px": 0.0,
            "total_crack_length_px": 0.0,
        }

    # Total crack area (total crack pixels)
    crack_area = int(np.sum(binary > 0))
    crack_density = round(crack_area / total_pixels * 100, 4)  # percentage
    crack_count = len(contours)

    # Estimate crack width per contour using bounding rect min dimension
    widths = []
    total_length = 0.0
    for contour in contours:
        if cv2.contourArea(contour) < 5:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        # Width is the smaller dimension of the bounding box
        widths.append(min(cw, ch))
        # Length approximated by the perimeter of the contour
        total_length += cv2.arcLength(contour, closed=False)

    mean_width = round(float(np.mean(widths)) if widths else 0.0, 4)
    total_length = round(total_length, 4)

    return {
        "crack_area_px": crack_area,
        "crack_density": crack_density,
        "crack_count": crack_count,
        "mean_crack_width_px": mean_width,
        "total_crack_length_px": total_length,
    }


def extract_features_from_folder(img_dir, mask_dir=None, unet_model_path=None, output_csv=None):
    """
    Runs feature extraction on all images in a folder.
    If mask_dir is given, uses pre-computed masks.
    Otherwise, runs U-Net inference to get masks.
    """
    if output_csv is None:
        output_csv = os.path.join("data", "processed", "features.csv")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    records = []
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:30]

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        base = os.path.splitext(img_file)[0]

        if mask_dir:
            mask_path = os.path.join(mask_dir, base + ".png")
            if not os.path.exists(mask_path):
                continue
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = run_unet_segmentation(img_path, unet_model_path)

        features = extract_crack_features(mask, img_path)
        features["image"] = img_file
        records.append(features)
        print(f"Processed: {img_file}")

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\nFeature extraction complete. Saved {len(df)} records to {output_csv}")
    return df


if __name__ == "__main__":
    img_dir = os.path.join("data", "segmentation", "deepcrack", "train_img")
    mask_dir = os.path.join("data", "segmentation", "deepcrack", "train_lab")

    print("Extracting features using pre-computed segmentation masks...")
    df = extract_features_from_folder(
        img_dir=img_dir,
        mask_dir=mask_dir,
        output_csv=os.path.join("data", "processed", "features.csv")
    )
    print(df.head())
