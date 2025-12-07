
---

# PatchCore Anomaly Inspector

## FastAPI • Docker • ResNet18 • PatchCore KNN • Web UI

This project implements a local industrial anomaly detection system based on PatchCore, using feature extraction with ResNet-18 and a precomputed memory bank.
It includes a complete FastAPI backend, a modern Web UI, IoU metrics, batch processing, real-time camera inspection, and Docker-based deployment.

## Main Features

* Core: Feature extraction using ResNet-18 and KNN-based neighborhood comparison.

* Web Interface: Dashboard for batch analysis, metrics (IoU, score), heatmaps, and real-time camera mode.

* API: Robust backend built with FastAPI.

* Advanced ROI: Support for exclusion masks and border margins.

* Logging: Automatic recording of predictions into a CSV file.

## Running the Project (Docker)
```
The project is containerized and configured to run automatically.

# 1. Build and start the service
docker compose up -d --build api

# 2. View real-time logs
docker compose logs -f api

# 3. Stop the service
docker compose down api
```

Access the Web UI at: http://localhost:8000.

## Project Structure
```
DP-Antolin-IA/
├── Backend/
│   ├── main.py                  # FastAPI backend + PatchCore pipeline
│   ├── models/patchcore/        # Memory bank + config.json
│   ├── static/
│   │   ├── assets/              # Frontend JS + CSS
│   │   └── overlays/            # Generated overlays and heatmaps
│   ├── templates/index.html     # Web UI
│   ├── logs/predictions.csv     # Auto-generated logs
│   └── .env                     # Backend configuration
│
├── Dataset/                     # Local dataset (optional)
├── notebooks/                   # Training & memory bank generation
├── docker-compose.yml
├── Dockerfile.cpu               # Default CPU-only Docker image
├── Backend_Documentation.md        
├── Deployment_Manual.txt
├── README_Dataset.md
├── User_Manual.pdf
└── README.md

```

## Environment Variables (.env)

The main model parameters are defined in Backend/.env. Key variables include:

* THRESHOLD: Decision threshold (default: 0.33).

* IMG_SIZE: Input resolution (default: 256).

* SAVE_VIS: Set to 1 to save debug images (overlays/heatmaps).

* ROI_PATH: Path to a PNG mask defining the inspection region.

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Loads the Web Interface (UI). |
| `GET` | `/health` | Service status and current configuration. |
| `POST` | `/predict` | Single-image inference (params: `thr`, `source`). |
| `POST` | `/predict_batch` | Processes multiple images and returns metrics. |

## Technical Architecture

1. Preprocessing: Image resizing and normalization.

2. Feature Extraction: Hooks in ResNet-18 layer2 and layer3.

3. PatchCore: Comparison with the memory bank using KNN ($K=3$).

4. Scoring: Maximum heatmap value restricted to the valid ROI.

5. Metrics: IoU calculation (Real IoU if ground truth mask is provided; approximate IoU otherwise).

## Web Interface (Frontend)
The Web UI (HTML + JS + CSS) provides:
* Image input: multiple file uploads, drag & drop.
* Threshold configuration: detection modes (normal, sensitive, strict).
* Results view: gallery view, table view, CSV export, KPIs dashboard.

## Real-Time Camera Inspection
The interface includes a live camera module that:
* Captures frames every 1.5 seconds.
* Displays: overlay heatmap, score, total defect area, state (NORMAL / ANOMALY), approximate IoU.

## Recall Metric
Users can manually assign the correct label after the first batch run:
* Each image includes two radio buttons: Normal / Defect.
* The ground-truth labels are sent in a second request.
* Recall is computed and shown in the KPI panel.

---
