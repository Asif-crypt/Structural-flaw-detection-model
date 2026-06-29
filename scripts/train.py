import argparse
import os
import shutil

import torch
from ultralytics import YOLO


def create_tiny_dataset():
    src_base = os.path.join("data", "yolo_dataset")
    tiny_base = os.path.join("data", "yolo_dataset_tiny")

    if os.path.exists(tiny_base):
        return

    print("Creating tiny dataset for fast verification...")
    for split, limit in [("train", 30), ("valid", 10)]:
        src_img_dir = os.path.join(src_base, split, "images")
        src_label_dir = os.path.join(src_base, split, "labels")
        dest_img_dir = os.path.join(tiny_base, split, "images")
        dest_label_dir = os.path.join(tiny_base, split, "labels")
        os.makedirs(dest_img_dir, exist_ok=True)
        os.makedirs(dest_label_dir, exist_ok=True)

        for image_name in os.listdir(src_img_dir)[:limit]:
            base_name, _ = os.path.splitext(image_name)
            label_name = base_name + ".txt"
            shutil.copy(os.path.join(src_img_dir, image_name), os.path.join(dest_img_dir, image_name))
            shutil.copy(os.path.join(src_label_dir, label_name), os.path.join(dest_label_dir, label_name))

    with open("dataset_tiny.yaml", "w", encoding="utf-8") as yaml_file:
        yaml_file.write(
            f"""path: {os.path.abspath(tiny_base)}
train: train/images
val: valid/images

names:
  0: crack
"""
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 crack detector.")
    parser.add_argument("--data", default="dataset.yaml", help="YOLO dataset YAML path.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers. Use 0 on Windows.")
    parser.add_argument("--model", default="yolov8n.pt", help="Starting YOLO weights.")
    parser.add_argument("--device", default=None, help="cuda device id, cpu, or auto when omitted.")
    parser.add_argument("--tiny", action="store_true", help="Train on the tiny smoke-test dataset.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.tiny:
        create_tiny_dataset()
        args.data = "dataset_tiny.yaml"
        args.epochs = min(args.epochs, 5)
        args.imgsz = min(args.imgsz, 320)
        args.batch = min(args.batch, 4)

    device = args.device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Training YOLOv8 on {args.data}")
    print(f"Device: {device}")

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project="runs/detect",
        name="train",
        exist_ok=True,
    )

    best_path = os.path.join("runs", "detect", "train", "weights", "best.pt")
    if os.path.exists(best_path):
        os.makedirs("models", exist_ok=True)
        shutil.copy(best_path, os.path.join("models", "best.pt"))
        print("Copied best YOLO weights to models/best.pt")

    print("YOLO training complete.")
