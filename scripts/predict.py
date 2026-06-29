import os
from ultralytics import YOLO

# Load the trained custom model weights
model_path = os.path.join("runs", "detect", "train", "weights", "best.pt")
if not os.path.exists(model_path):
    # Fallback to copy in models folder if we moved it
    model_path = os.path.join("models", "best.pt")

model = YOLO(model_path)

# Predict on the test image
test_img = "data/crack_detection/test.jpg"
results = model.predict(
    source=test_img,
    save=True,
    conf=0.01 # Since it's trained for only 3 epochs, confidence might be low
)

print("Prediction Complete")
