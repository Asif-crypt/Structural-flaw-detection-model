from ultralytics import YOLO
import os
import shutil

# Load a pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")

# Set up test image path
test_img = "data/crack_detection/test.jpg"

if not os.path.exists(test_img):
    # Fallback to look for a sample image in the positive Kaggle dataset
    pos_dir = os.path.join("data", "crack_detection", "Positive")
    if os.path.exists(pos_dir) and os.listdir(pos_dir):
        first_img = os.path.join(pos_dir, os.listdir(pos_dir)[0])
        shutil.copy(first_img, test_img)
        print(f"Copied test image from {first_img} to {test_img}")
    else:
        print("No positive crack images found to copy. Please place an image at data/crack_detection/test.jpg manually.")

# Run inference
if os.path.exists(test_img):
    results = model.predict(
        source=test_img,
        save=True
    )
    print("Detection Complete")
else:
    print(f"Test image not found at {test_img}. Cannot run prediction.")
