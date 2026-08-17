# 🏗️ StructuralAI: AI-Powered Structural Health Monitoring System

StructuralAI is a multi-modal Artificial Intelligence framework designed to detect, analyze, and monitor structural flaws in civil infrastructure (such as concrete buildings, bridges, and tunnels). By combining **Computer Vision (YOLOv8 & U-Net)** for visual inspections and **Tabular Machine Learning (XGBoost & Random Forest)** for sensor-based health prediction, the platform offers a comprehensive toolset for structural engineers.

Additionally, the system features **Explainable AI (SHAP)** to clarify model predictions and a **LangChain & Gemini-powered Chat Agent** that generates inspection reports and provides interactive guidance.

---

## 🚀 Key Features

*   🔍 **Multi-Modal Visual Inspection**:
    *   **YOLOv8**: Detects structural cracks with bounding boxes and confidence scores.
    *   **Tiny U-Net**: Performs pixel-level semantic crack segmentation to measure geometry, estimate maximum crack depth, and compute crack density.
    *   **Thermal HSV Anomaly Extraction**: Detects moisture, water leaks, and heat bridges in infrared thermal images.
*   📊 **Tabular Risk & Health Assessment**:
    *   Predicts **Structural Health Index (SHI)** using XGBoost Regressor.
    *   Classifies **Risk Level** (Safe, Moderate, Critical) using XGBoost Classifier.
    *   Estimates **Remaining Useful Life (RUL)** using Random Forest Regressor.
    *   Calculates automated structural repair cost ranges.
*   🧠 **Explainable AI (SHAP)**:
    *   Generates dynamic waterfall plots using Shapley values to illustrate the exact feature contributions behind every SHI prediction.
*   📝 **Dynamic Reports & AI Copilot**:
    *   Generates a professional, print-ready PDF engineering inspection report complete with SHAP plots and annotated crack images.
    *   Includes a conversational **AI Inspector Chatbot** powered by Google Gemini and LangChain, capable of answering queries based on the inspection report context.
*   💻 **Dual Client Interfaces**:
    *   **React Dashboard**: A modern, high-fidelity single-page application built with React.
    *   **Streamlit Web App**: A lightweight, Python-native interface for quick prototyping.

---

## 📂 Project Structure

```text
Structural flaw detection model/
├── backend/
│   └── main.py                     # FastAPI REST API (inference, report gen, chatbot)
├── frontend-react/                 # Modern React Dashboard Single-Page Application
│   ├── public/
│   ├── src/
│   │   ├── components/             # UI components (Dashboard, CrackDetection, RiskPredictor, etc.)
│   │   ├── services/
│   │   │   └── api.js              # Axios-based API client for Backend REST API
│   │   ├── App.js
│   │   └── index.js
│   └── package.json
├── scripts/                        # Model training, data preparation, & utility scripts
│   ├── agent.py                    # LangChain & Gemini LLM report summary & chat logic
│   ├── detect.py                   # YOLOv8 batch detector script
│   ├── explain.py                  # Standalone SHAP explainer utility
│   ├── extract_features.py         # Visual & tabular feature fusion helper
│   ├── generate_report.py          # ReportLab PDF building script
│   ├── generate_tabular_data.py    # Synthetic tabular dataset generator
│   ├── prepare_yolo_dataset.py     # Converts raw imagery for YOLO training
│   ├── train.py                    # YOLOv8 trainer
│   ├── train_ml.py                 # XGBoost / Random Forest tabular models trainer
│   └── train_unet.py               # U-Net semantic segmentation trainer
├── reports/                        # Auto-generated report output directory (PDFs & temporary plots)
├── app.py                          # Streamlit front-end application
├── main.py                         # Root entry point to launch FastAPI backend
├── requirements.txt                # Python backend dependencies
├── dataset.yaml                    # YOLOv8 main dataset configuration
└── dataset_tiny.yaml               # YOLOv8 tiny dataset configuration
```

---

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.9 or higher
*   Node.js (v16+) and npm

### 1. Set Up the Python Backend

First, navigate to the repository directory and set up a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory to enable the AI Report Analyst and Inspector Chatbot:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
STRUCTURALAI_API_URL=http://127.0.0.1:8000
```

### 3. Generate Synthetic Data & Train the Models

Since model weights (`.pt`, `.pth`, `.pkl`) and synthetic data are ignored by git, train them locally to run the application:

```bash
# Generate synthetic tabular training data (saved to data/tabular/)
python scripts/generate_tabular_data.py

# Train the tabular models (SHI, Risk Level, RUL)
python scripts/train_ml.py

# Train the U-Net crack segmentation model
python scripts/train_unet.py

# Train the YOLOv8 crack detection model (optional, requires raw images)
python scripts/train.py
```

### 4. Running the Servers

#### Launch the FastAPI Backend:
```bash
python main.py
```
The API server will launch at `http://127.0.0.1:8000`. You can inspect the interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

#### Run the React Frontend:
In a new terminal window, navigate to the React directory and launch the client application:
```bash
cd frontend-react
npm install
npm start
```
This launches the React development server at `http://localhost:3000`.

#### Run the Streamlit Frontend (Alternative):
If you prefer to run the Streamlit app:
```bash
streamlit run app.py
```
This launches Streamlit at `http://localhost:8501`.

---

## 🔌 API Endpoint Documentation

The FastAPI backend exposes the following endpoints:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | API root check (returns running state and docs URL). |
| `/health` | `GET` | Basic health/uptime check. |
| `/detect` | `POST` | Crack detection on uploaded images using YOLOv8. |
| `/analyze_image` | `POST` | Full visual analysis: YOLOv8 detection + U-Net pixel segmentation. Returns base64 annotated images, segmentation masks, crack area, density, and depth. |
| `/analyze_thermal` | `POST` | HSV thermal anomaly detection. Identifies moisture/leaks and heat bridges. |
| `/predict_risk` | `POST` | Takes structural sensor readings and outputs SHI, Risk classification, RUL, and repair cost ranges. |
| `/explain_risk` | `POST` | Explains ML risk metrics by generating and returning a base64 SHAP waterfall plot. |
| `/analyze_structure` | `POST` | Executes the LangChain engineering recommendation engine. |
| `/generate_report` | `POST` | Generates and serves a downloadable PDF inspection report containing SHAP plots, annotated visual logs, and LLM engineering suggestions. |
| `/chat` | `POST` | Converses with the AI inspector chatbot, utilizing inspection report context. |

---

## 📊 Tabular Features Reference

When inputting data for risk/health prediction (via React UI, Streamlit, or raw API JSON), the following features are processed:

*   `building_age` (years): The age of the target structure.
*   `corrosion_level` (0.0 to 1.0): Level of reinforcing bar steel corrosion.
*   `crack_width` (mm): Average crack opening width.
*   `crack_density` (cracks/m²): Number/density of visual cracks.
*   `moisture_content` (%): Relative internal moisture readings.
*   `compressive_strength` (MPa): Concrete core compressive strength.
*   `temperature` (°C): Ambient sensor temperature.
*   `humidity` (%): Ambient relative humidity.
*   `load_stress` (MPa): Dynamic stress/strain from structural loads.
