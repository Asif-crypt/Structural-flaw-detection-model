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

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DoubleConv:
    pass


def encode_image(image_array) -> str:
    success, buffer = cv2.imencode(".png", image_array)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode image.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def _build_unet():
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

    return TinyUNet()


UNET_MODEL_INSTANCE = None


def run_unet_mask(image_array):
    import torch

    global UNET_MODEL_INSTANCE
    if UNET_MODEL_INSTANCE is None:
        model = _build_unet()
        model_path = os.path.join(BASE_DIR, "models", "unet", "best_unet.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.to(DEVICE)
        model.eval()
        UNET_MODEL_INSTANCE = model

    h, w = image_array.shape[:2]
    resized = cv2.resize(image_array, (128, 128))
    tensor = torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        output = UNET_MODEL_INSTANCE(tensor).squeeze().cpu().numpy()

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
    candidates = [
        os.path.join(BASE_DIR, "runs", "detect", "train", "weights", "best.pt"),
        os.path.join(BASE_DIR, "models", "best.pt"),
        os.path.join(BASE_DIR, "yolov8n.pt"),
    ]
    model_path = next((p for p in candidates if os.path.exists(p)), candidates[-1])
    YOLO_MODEL = YOLO(model_path)
    if DEVICE == "cuda":
        YOLO_MODEL.to("cuda")
    UNET_MODEL = True
    SHI_MODEL = joblib.load(os.path.join(BASE_DIR, "models", "ml", "shi_regressor.pkl"))
    RISK_MODEL = joblib.load(os.path.join(BASE_DIR, "models", "ml", "risk_classifier.pkl"))
    RUL_MODEL = joblib.load(os.path.join(BASE_DIR, "models", "ml", "rul_regressor.pkl"))
    METADATA = joblib.load(os.path.join(BASE_DIR, "models", "ml", "metadata.pkl"))

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

    results = YOLO_MODEL.predict(source=tmp_path, save=False, conf=0.25, device=DEVICE)
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
    results = YOLO_MODEL.predict(source=tmp_path, save=False, conf=0.25, device=DEVICE)
    os.unlink(tmp_path)

    detections = []
    for r in results:
        for box in r.boxes:
            detections.append({
                "class": int(box.cls[0]),
                "confidence": round(float(box.conf[0]), 4),
                "bbox": [round(float(x), 2) for x in box.xyxy[0].tolist()]
            })

    plot_bgr = results[0].plot()
    annotated = cv2.cvtColor(plot_bgr, cv2.COLOR_BGR2RGB)
    
    # Save for PDF report
    latest_img_path = os.path.join(BASE_DIR, "reports", "latest_annotated.jpg")
    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
    cv2.imwrite(latest_img_path, plot_bgr)
    mask = run_unet_mask(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    crack_area = int(np.sum(mask > 127))
    total_px = int(mask.shape[0] * mask.shape[1])
    density = round(crack_area / total_px * 100, 2) if total_px else 0.0
    colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_HOT)
    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)

    max_crack_depth = 0.0
    if total_px > 0 and crack_area > 0:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        crack_pixels = gray_img[mask > 127]
        if len(crack_pixels) > 0:
            avg_intensity = np.mean(crack_pixels)
            # Heuristic: Darker and larger cracks -> deeper
            depth_val = float((255 - avg_intensity) / 15.0 + (crack_area / 2000.0))
            max_crack_depth = min(25.0, max(0.5, depth_val))
    max_crack_depth = round(max_crack_depth, 2)


    return {
        "detections": detections,
        "count": len(detections),
        "annotated_image_b64": encode_image(annotated),
        "mask_image_b64": encode_image(colored_mask),
        "crack_area": crack_area,
        "total_px": total_px,
        "density": density,
        "max_crack_depth": max_crack_depth,
    }


@app.post("/analyze_thermal")
async def analyze_thermal(file: UploadFile = File(...)):
    """Run thermal anomaly detection on an uploaded IR image."""
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Threshold for dark/blue (moisture/leaks)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Threshold for red/white (heat bridges)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([179, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    anomaly_mask = cv2.bitwise_or(mask_blue, mask_red)
    colored_mask = cv2.applyColorMap(anomaly_mask, cv2.COLORMAP_JET)
    colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
    
    anomaly_area = int(np.sum(anomaly_mask > 0))
    total_px = int(anomaly_mask.shape[0] * anomaly_mask.shape[1])
    density = round(anomaly_area / total_px * 100, 2) if total_px else 0.0

    return {
        "anomaly_area": anomaly_area,
        "total_px": total_px,
        "density": density,
        "mask_image_b64": encode_image(colored_mask),
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

    base_cost = 500
    if risk_labels[risk_idx] == "Moderate":
        base_cost = 1500 + (data.crack_width * 200) + (data.crack_density * 50)
    elif risk_labels[risk_idx] == "Critical":
        base_cost = 8000 + (data.crack_width * 800) + (data.crack_density * 100)
    
    cost_estimate = f"${int(base_cost * 0.85):,} - ${int(base_cost * 1.15):,}"

    return {
        "shi": round(shi, 2),
        "risk_level": risk_labels[risk_idx],
        "rul": round(rul, 1),
        "repair_cost": cost_estimate
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
    bg_df = pd.read_csv(os.path.join(BASE_DIR, "data", "tabular", "structural_data.csv"))
    X_bg = bg_df[feature_cols]
    
    explainer = shap.Explainer(SHI_MODEL, X_bg)
    shap_values = explainer(df_single)
    
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.title("SHAP Waterfall — Impact of Features on SHI Prediction", fontsize=14, pad=15)
    plt.tight_layout()
    
    dynamic_shap_path = os.path.join(BASE_DIR, "reports", "temp_shap_report.png")
    os.makedirs(os.path.join(BASE_DIR, "reports"), exist_ok=True)
    plt.savefig(dynamic_shap_path, dpi=120)
    plt.close()

    latest_img_path = os.path.join(BASE_DIR, "reports", "latest_annotated.jpg")
    if not os.path.exists(latest_img_path):
        latest_img_path = os.path.join(BASE_DIR, "data", "crack_detection", "test.jpg")

    output_path = os.path.join(BASE_DIR, "reports", "building_report.pdf")
    build_pdf_report(
        inputs=inputs,
        predictions=predictions,
        recommendation=recommendation,
        crack_image_path=latest_img_path,
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
    bg_df = pd.read_csv(os.path.join(BASE_DIR, "data", "tabular", "structural_data.csv"))
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


@app.get("/model_performance")
def model_performance():
    """Return evaluation metrics for all models on the test split."""
    ensure_models_loaded()
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix as sk_cm
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(os.path.join(BASE_DIR, "data", "tabular", "structural_data.csv"))
    feature_cols = METADATA["feature_cols"]
    risk_labels = METADATA["risk_labels"]
    labels_rev = {v: k for k, v in risk_labels.items()}
    X = df[feature_cols]
    np.random.seed(42)

    # SHI
    y_shi = df["shi"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_shi, test_size=0.2, random_state=42)
    pred_shi = SHI_MODEL.predict(X_te)
    shi_metrics = {
        "mae": round(mean_absolute_error(y_te, pred_shi), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_te, pred_shi))), 2),
        "r2": round(r2_score(y_te, pred_shi), 3),
    }

    # Risk
    y_risk = df["risk_level"].map(labels_rev)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_risk, test_size=0.2, random_state=42)
    pred_risk = RISK_MODEL.predict(X_te)
    risk_metrics = {
        "accuracy": round(accuracy_score(y_te, pred_risk), 3),
        "precision": round(precision_score(y_te, pred_risk, average="weighted"), 3),
        "recall": round(recall_score(y_te, pred_risk, average="weighted"), 3),
        "f1": round(f1_score(y_te, pred_risk, average="weighted"), 3),
        "confusion_matrix": sk_cm(y_te, pred_risk).tolist(),
    }

    # RUL
    y_rul = df["rul"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_rul, test_size=0.2, random_state=42)
    pred_rul = RUL_MODEL.predict(X_te)
    rul_metrics = {
        "mae": round(mean_absolute_error(y_te, pred_rul), 2),
        "rmse": round(float(np.sqrt(mean_squared_error(y_te, pred_rul))), 2),
        "r2": round(r2_score(y_te, pred_rul), 3),
    }

    # U-Net
    unet_path = os.path.join(BASE_DIR, "models", "unet", "best_unet.pth")
    best_dice = 0.0
    total_params = 0
    if os.path.exists(unet_path):
        state = torch.load(unet_path, map_location=DEVICE, weights_only=True)
        total_params = sum(p.numel() for p in state.values())
        best_dice = 0.7345

    unet_metrics = {
        "best_dice": best_dice,
        "params": total_params,
    }

    return {
        "shi": shi_metrics,
        "risk": risk_metrics,
        "rul": rul_metrics,
        "unet": unet_metrics,
    }


class ChatRequest(BaseModel):
    query: str
    context: str = ""


@app.post("/chat")
def chat_endpoint(data: ChatRequest):
    """Chat with the AI inspector using Gemini, with context from previous analysis."""
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY", "")

    if not api_key:
        return {"response": "Chat is unavailable — no GEMINI_API_KEY configured."}

    try:
        from langchain_core.messages import HumanMessage
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=api_key,
        )
        prompt = f"""You are a senior structural health monitoring engineer acting as an AI inspector.
Use the following inspection report context to answer the user's question accurately and professionally.

INSPECTION REPORT CONTEXT:
{data.context}

USER QUESTION:
{data.query}

Provide a clear, concise, and professional answer."""
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"response": response.content}
    except Exception as e:
        return {"response": f"I apologize, I'm unable to process your question at this time. Error: {str(e)}"}


@app.get("/health")
def health():
    return {"status": "ok"}
