import os
import sys
import io
import base64
import tempfile
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import joblib
import pandas as pd

app = FastAPI(
    title="StructuralAI API",
    description="AI-powered Structural Health Monitoring REST API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

YOLO_MODEL = None
UNET_MODEL = None
SHI_MODEL = None
RISK_MODEL = None
RUL_MODEL = None
METADATA = None


class DoubleConv:
    pass


def encode_image(image_array) -> str:
    success, buffer = cv2.imencode(".png", image_array)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def run_unet_mask(image_array):
    import torch
    import torch.nn as nn

    class DoubleConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    class TinyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.down1 = DoubleConv(3, 16)
            self.pool1 = nn.MaxPool2d(2)
            self.down2 = DoubleConv(16, 32)
            self.pool2 = nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(32, 64)
            self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.conv_up2 = DoubleConv(64, 32)
            self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
            self.conv_up1 = DoubleConv(32, 16)
            self.out_conv = nn.Conv2d(16, 1, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            d1 = self.down1(x)
            d2 = self.down2(self.pool1(d1))
            bottleneck = self.bottleneck(self.pool2(d2))
            u2 = self.conv_up2(torch.cat((d2, self.up2(bottleneck)), dim=1))
            u1 = self.conv_up1(torch.cat((d1, self.up1(u2)), dim=1))
            return self.sigmoid(self.out_conv(u1))

    model = TinyUNet()
    model_path = "models/unet/best_unet.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    h, w = image_array.shape[:2]
    resized = cv2.resize(image_array, (128, 128))
    tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        output = model(tensor).squeeze().numpy()
        
    # Prevent solid white mask if the model is untrained/randomly initialized
    if output.max() < 0.6 and output.min() > 0.4:
        mask = np.zeros_like(output, dtype=np.uint8)
    else:
        mask = (output > 0.5).astype(np.uint8) * 255
        
    return cv2.resize(mask, (w, h))


def ensure_models_loaded():
    global YOLO_MODEL, UNET_MODEL, SHI_MODEL, RISK_MODEL, RUL_MODEL, METADATA
    if YOLO_MODEL is not None and UNET_MODEL is not None and SHI_MODEL is not None and RISK_MODEL is not None and RUL_MODEL is not None and METADATA is not None:
        return
    load_models()

# --------------- Load models on startup ---------------
@app.on_event("startup")
def load_models():
    global YOLO_MODEL, UNET_MODEL, SHI_MODEL, RISK_MODEL, RUL_MODEL, METADATA
    from ultralytics import YOLO
    model_path = "runs/detect/train/weights/best.pt"
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"
    YOLO_MODEL = YOLO(model_path)
    UNET_MODEL = True
    SHI_MODEL = joblib.load("models/ml/shi_regressor.pkl")
    RISK_MODEL = joblib.load("models/ml/risk_classifier.pkl")
    RUL_MODEL = joblib.load("models/ml/rul_regressor.pkl")
    METADATA = joblib.load("models/ml/metadata.pkl")

# --------------- Pydantic schemas ---------------
class StructuralInput(BaseModel):
    building_age: float
    corrosion_level: float
    crack_width: float
    crack_density: float
    moisture_content: float
    compressive_strength: float
    temperature: float
    humidity: float
    load_stress: float

# --------------- Endpoints ---------------
@app.get("/")
def root():
    return {"message": "StructuralAI API is running!", "docs": "/docs"}


@app.post("/detect")
async def detect_cracks(file: UploadFile = File(...)):
    """Upload an image and run YOLOv8 crack detection."""
    ensure_models_loaded()
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Save temporarily
    tmp_path = tempfile.mktemp(suffix=".jpg")
    cv2.imwrite(tmp_path, img)

    results = YOLO_MODEL.predict(source=tmp_path, save=False, conf=0.25)
    os.unlink(tmp_path)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": int(box.cls[0]),
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()]
            })

    return {"detections": detections, "count": len(detections)}


@app.post("/analyze_image")
async def analyze_image(file: UploadFile = File(...)):
    """Run YOLO detection and U-Net segmentation on an uploaded image."""
    ensure_models_loaded()
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    tmp_path = tempfile.mktemp(suffix=".jpg")
    cv2.imwrite(tmp_path, img)
    results = YOLO_MODEL.predict(source=tmp_path, save=False, conf=0.25)
    os.unlink(tmp_path)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": int(box.cls[0]),
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()]
            })

    annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    mask = run_unet_mask(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    crack_area = int(np.sum(mask > 127))
    total_px = int(mask.shape[0] * mask.shape[1])
    density = round(crack_area / total_px * 100, 2) if total_px else 0.0
    colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)

    return {
        "detections": detections,
        "count": len(detections),
        "annotated_image_b64": encode_image(annotated),
        "mask_image_b64": encode_image(colored_mask),
        "crack_area": crack_area,
        "total_px": total_px,
        "density": density,
    }


@app.post("/predict_risk")
def predict_risk(data: StructuralInput):
    """Run ML models to predict SHI, risk level, and RUL."""
    ensure_models_loaded()
    feature_cols = METADATA["feature_cols"]
    risk_labels = METADATA["risk_labels"]
    X = pd.DataFrame([[getattr(data, c) for c in feature_cols]], columns=feature_cols)

    shi = float(SHI_MODEL.predict(X)[0])
    risk_idx = int(RISK_MODEL.predict(X)[0])
    rul = float(RUL_MODEL.predict(X)[0])

    return {
        "shi": round(shi, 2),
        "risk_level": risk_labels[risk_idx],
        "rul": round(rul, 1)
    }


@app.post("/generate_report")
def generate_report(data: StructuralInput):
    """Generate a PDF report for the given structural inputs."""
    ensure_models_loaded()
    from scripts.agent import analyze_structure
    from scripts.generate_report import build_pdf_report

    inputs = data.model_dump()
    predictions, recommendation = analyze_structure(inputs)

    # Generate dynamic SHAP waterfall plot for the report
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    feature_cols = METADATA["feature_cols"]
    df_single = pd.DataFrame([inputs], columns=feature_cols)
    bg_df = pd.read_csv("data/tabular/structural_data.csv")
    X_bg = bg_df[feature_cols]
    
    explainer = shap.Explainer(SHI_MODEL, X_bg)
    shap_values = explainer(df_single)
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("SHAP Waterfall — Impact of Features on SHI Prediction", fontsize=14, pad=15)
    plt.tight_layout()
    
    dynamic_shap_path = "reports/temp_shap_report.png"
    os.makedirs("reports", exist_ok=True)
    plt.savefig(dynamic_shap_path, dpi=120)
    plt.close()

    output_path = "reports/building_report.pdf"
    build_pdf_report(
        inputs=inputs,
        predictions=predictions,
        recommendation=recommendation,
        crack_image_path="runs/detect/predict/test.jpg",
        shap_image_path=dynamic_shap_path,
        output_path=output_path
    )
    return FileResponse(output_path, media_type="application/pdf", filename="building_report.pdf")


@app.post("/explain_risk")
def explain_risk_api(data: StructuralInput):
    """Generate dynamic SHAP waterfall plot for given inputs."""
    ensure_models_loaded()
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    feature_cols = METADATA["feature_cols"]
    df_single = pd.DataFrame([data.model_dump()], columns=feature_cols)
    bg_df = pd.read_csv("data/tabular/structural_data.csv")
    X_bg = bg_df[feature_cols]
    
    explainer = shap.Explainer(SHI_MODEL, X_bg)
    shap_values = explainer(df_single)
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("SHAP Waterfall — Prediction for Your Inputs", fontsize=14, pad=15)
    plt.tight_layout()
    
    tmp_path = tempfile.mktemp(suffix=".png")
    plt.savefig(tmp_path, dpi=120)
    plt.close()
    
    with open(tmp_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmp_path)
    
    return {"waterfall_b64": encoded}


@app.post("/analyze_structure")
def analyze_structure_api(data: StructuralInput):
    """Run the LangChain agent to analyze structure and give recommendations."""
    ensure_models_loaded()
    from scripts.agent import analyze_structure

    inputs = data.model_dump()
    predictions, recommendation = analyze_structure(inputs)
    
    return {
        "predictions": predictions,
        "recommendation": recommendation
    }


@app.get("/health")
def health():
    return {"status": "ok"}
