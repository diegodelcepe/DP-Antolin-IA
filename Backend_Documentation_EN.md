# Complete Backend Documentation (Web Application + Anomaly Inference API)

## 1. General Introduction
This project implements a complete system for `anomaly detection` in industrial parts using computer vision.
It consists of an *`inference backend`* based on PatchCore and a specialized *`dataset`* that trains and feeds this model.

## 1.1 Backend - Anomaly Inference API
This backend implements an **anomaly inference API** based on **PatchCore** over images, built with **FastAPI**.  
It exposes endpoints to check system status, serve a static frontend, and perform anomaly predictions.

Internally, it loads a **ResNet18 backbone**, a **memory bank** (KNN over normalized embeddings), and optionally applies a **Region of Interest (ROI)** along with visualization operations (overlays and polygons).

>**Note:** An inference API is an interface that allows sending data (in this case, images) to a pre-trained artificial intelligence model and receiving results or predictions.

### Main Objectives
- Receive an image.  
- Transform it and extract features.  
- Calculate an anomaly heatmap and score.  
- Decide if it's anomalous based on flexible thresholds.  
- *(Optional)* Generate visualizations and polygons of anomalous areas for the frontend.

## 1.2 Dataset
The dataset used is called `dataset_gua_crops`, containing real images of plastic parts made by *`Antolin`* for inspection tasks. This dataset is the core that feeds the PatchCore model.

### Dataset Objective

The purpose of the dataset is to train and validate the model, so it can learn the normal behavior of the parts and detect any deviation as an anomaly. Thus, the system contributes to:
* Reduce manufacturing waste.
* Improve quality control.

---

## 2. General Architecture
This section details how the internal components of the system are organized and how they interact with each other.

**Key Components:**
- **FastAPI** → HTTP server that receives requests (e.g., `/predict`) and also automatically generates API documentation (Swagger/OpenAPI).  
- **Inference module (`main.py`)** → Core of the backend: receives the image, processes it, executes the model, and returns the JSON response.  
- **Model artifacts** → Files that store the system's knowledge, located in `Backend/models/patchcore` (according to `ARTIFACTS_DIR` variable):  
  - `memory_bank_core.npz`: contains embeddings of normal images (the "memory" of what is normal).  
  - `config.json`: stores parameters such as threshold (`threshold`) or embedding version.  
- **`static/` folder** → Contains static resources and generated visualizations (overlays, heatmaps, masks).  
- **`templates/` folder** → Includes `index.html`, a minimal frontend served at the root `/`.  
- **Optional ROI** → Can be defined via a PNG mask (`ROI_PATH`) or a percentage crop (`IGNORE_BORDER_PCT`) to limit the region used in score calculation.  
- **Precalculated KNN** → The system compares each patch of the image with the *memory bank* embeddings using a KNN algorithm (k nearest neighbors) to calculate rarity distances.  
- **PyTorch hooks** → Hook into intermediate layers of ResNet18 (`layer2` and `layer3`) to extract multi-scale representations (textures and shapes) and fuse them into a joint representation.  

**Summarized Flow:**

```draw.io
[Client/Frontend] --> /predict (POST, image)
        |
        v
 [Image reading and normalization]
        |
        v
 [Feature extraction (ResNet18 + hooks)]
        |
        v
 [Patchify + Normalization + KNN Distances]
        |
        v
 [Heatmap + Normalization (0..1)]
        |
        v
 [ROI application (optional) for score]
        |
        v
 [Maximum score calculation and comparison with threshold]
        |
        v
 [Visualization generation (overlay/mask/polygons) if SAVE_VIS]
        |
        v
 [JSON response (score, threshold, is_anomaly, polygons, overlay_url)]
```

---

## 3. Directory Structure
This section explains how the source code is distributed within the project. Each folder and file has a clear purpose within the backend (model loading, visualization, templates, etc.).

```draw.io
Backend/
 ├─ main.py                # FastAPI core + inference logic
 ├─ requirements.txt       # Environment dependencies
 ├─ models/                # Artifacts (memory bank, config.json, etc.)
 │   └─ patchcore/         # Expected subfolder (by default)
 ├─ static/                # Static resources + generated overlays (/static/overlays)
 ├─ templates/             # index.html served at "/"
 └─ tests/                 # Tests (structure to expand)
```

**Relationship:**
- `main.py` mounts `/static` and serves `templates/index.html` at the root `/`.  
- The memory bank is loaded from `models/patchcore/memory_bank_core.npz`.  
- Visualizations are saved in `static/overlays/`.  

This modular organization facilitates maintenance, reproducibility, and integration with frontend and production deployments.

---

## 4. Configuration and Environment Variables
This section shows the different variables that allow adapting the `backend` behavior without needing to modify the code. These configurations control aspects like model sensitivity, visualization generation, ROI mask usage, and processing limits.

**Variables (with default values if not in `.env`):**
- `ARTIFACTS_DIR` (default: `models/patchcore`)
- `STATIC_DIR` (default: `static`)
- `OVERLAYS_SUBDIR` (default: `overlays`)
- `THRESHOLD` (default: `config.json.threshold` or 0.35 if not defined)
- `IMG_SIZE` (default: 256)
- `KNN_K` (default: 3)
- `PATCH_STRIDE` (default: 1)
- `SAVE_VIS` (`"1"` enables visualizations; `"0"` returns only JSON)  
- `AREA_MIN` (default: 200, minimum contour area)
- `IGNORE_BORDER_PCT` (percentage symmetrically cropped from each edge for ROI)
- `ROI_PATH` (path to binary PNG for ROI mask, size equal to `IMG_SIZE`)  

**Impact Summary:**
- Adjust sensitivity and computational cost.  
- Enable or disable visualizations.  
- Control which part of the image enters score calculation (ROI).  
- Allow threshold modification without code changes.

**Example `.env`:**
``` yaml
THRESHOLD=0.42
IMG_SIZE=256
KNN_K=5
SAVE_VIS=1
IGNORE_BORDER_PCT=8
ROI_PATH=./models/roi_mask.png
```

These variables allow flexible and reproducible system adjustment without modifying the source code.

---

## 5. Detailed Inference Flow
This section delves into the entire journey an image follows within the inference system. From when it's received and transformed, to obtaining the anomaly map and the final result. The calculations and operations that allow the backend to decide if an image presents an anomaly are explained step by step.

**Steps:**
1. **File loading and validation**: reception (`UploadFile`), MIME type/size verification, and decoding with OpenCV (BGR handling, conversion from BGRA or grayscale).  
2. **Basic preprocessing**: conversion to grayscale and resizing to `IMG_SIZE`.  
3. **Input tensor**: normalization to expected range and replication to 3 channels (ResNet18 expects 3 channels).  
4. **Backbone forward**: execution of ResNet18 with hooks on `layer2` and `layer3`.  
5. **Spatial alignment**: interpolation of `layer3` to match size to `layer2`.  
6. **Feature fusion**: channel concatenation `fcat = [layer2, layer3_up]`.  
7. **Patchify**: conversion of each spatial location into a vector; optional `stride` for subsampling.  
8. **L2 normalization per patch**: ensure comparability with memory.  
9. **KNN query**: each patch against the memory bank (k neighbors).  
10. **Distance map**: average distance to the `k` neighbors → rarity map.  
11. **Upsampling**: resizing of the map to `IMG_SIZE` resolution.  
12. **Normalization for visualization**: min-max of the map for overlay and heatmap (does not alter score calculation).  
13. **Score calculation**: maximum value of `heat` within the ROI (if defined); otherwise global maximum.  
14. **Decision**: comparison `score` vs effective `threshold` (base `.env` adjusted by `thr` and `mode`).  
15. **Visualization (optional)**: if `SAVE_VIS=1`, generation of overlay, colored heatmap, binary mask; morphological operations and extraction/approximation of contours and polygons.  
16. **Response**: JSON with `score`, `threshold`, `is_anomaly`, `polygons` (if anomaly) and `overlay_url`.

``` draw.io
+-------------------+
|   Input Image     |
|   (UploadFile)    |
+---------+---------+
          |
          v
+-------------------+
| Preprocessing     |
| - OpenCV decode   |
| - Gray + resize   |
| - Tensor 3 channels|
+---------+---------+
          |
          v
+-------------------+
| Backbone ResNet18 |
| Hooks: layer2/l3  |
+---------+---------+
          |
          v
+-------------------+
| Alignment & Fusion|
| - Upsample layer3  |
| - Channel concat   |
+---------+---------+
          |
          v
+-------------------+
| Patchify + L2 norm|
+---------+---------+
          |
          v
+-------------------+
| KNN Memory Bank   |
| - k distances     |
| - Rarity map      |
+---------+---------+
          |
          v
+-------------------+
| Postprocessing    |
| - Upsample map    |
| - Min-max norm    |
| - ROI + max score |
+---------+---------+
          |
          v
+-------------------+
| Comparison        |
| score vs threshold|
| (thr/mode/.env)   |
+---------+---------+
          |
          v
+-------------------+
| Visualization     |
| - Overlay/heatmap |
| - Mask/polygons   |
+---------+---------+
          |
          v
+-------------------+
| JSON Response     |
| score, threshold, |
| is_anomaly, polys,|
| overlay_url       |
+-------------------+
```

---

## 6. Backbone and Feature Extraction
This point describes the heart of the model: the ResNet18 backbone. It explains how its intermediate layers (hooks) are leveraged, how extracted features are combined, and why a KNN distance-based approach on embeddings is used. The goal is to understand how the system "learns" to recognize normality and detect what deviates from that pattern.

>**Note:** The backbone (ResNet18) acts as a feature extractor, generating visual representations at multiple abstraction levels (edges, textures, shapes), which serve as the basis for subsequent anomaly detection processes.

- **Backbone**: `ResNet18` pre-trained on ImageNet.  
- **Hooks**:
  - `layer2` captures mid-level features (texture, local edges).
  - `layer3` deeper features (shapes and global semantics); *upsample* is applied for spatial alignment.
- **Fusion**: channel concatenation ⇒ multi-scale representation combining fine detail and global context.
- **Patch normalization**: ensures that new embeddings have a distribution comparable to that stored in the *memory bank*, avoiding arbitrary scales.
- **KNN**: calculates average distances to the *k* nearest neighbors as a rarity score (anomaly = embedding not similar to the bank).

**Advantages:**
- Does not require retraining for each normal class (pre-built memory bank).  
- Scalable to different objects if the bank is rebuilt.

This design allows the system to learn normality in an unsupervised manner and detect deviations without needing to train a specific classifier.

---

## 7. ROI and Border Handling
This section describes the handling of `Regions of Interest (ROI)`. The goal is to allow the system to focus on relevant areas of the image, ignoring borders or irrelevant zones, in order to reduce false positives in anomaly detection.

**Available mechanisms:**
1. **Border cropping (`IGNORE_BORDER_PCT`)**: creates an ignored margin. Pixels in that zone are marked as 0 in the mask.
2. **External mask (`ROI_PATH`)**: binary image (white = valid area). The mask is rescaled to `IMG_SIZE` and combined with border cropping.

**Usage:**
- When calculating the score, pixels outside ROI are penalized (assigned a minimum value of `-1`).
- In visualization, the ROI border is drawn in cyan (0,255,255).
- The mask only affects score and binarization for polygons, not the colormap.

**Considerations:**
- If the final mask ends up all zero, ROI is ignored (returns `None`).
- Avoids false positives in irrelevant areas (borders, background).

This way, the system focuses only on relevant regions, improving the accuracy of anomaly detection.

---

## 8. Anomaly Map and Score Calculation
Here, the mathematical logic behind the result is explained. It details how the anomaly map (heatmap) is built, how values are normalized, and how a "score" is obtained that summarizes the image's rarity. It also describes how "sensitive" and "strict" modes dynamically adjust thresholds.

**Definitions:**
- `heat`: float32 map resulting from resizing patch distances.
- `hmin`, `hmax`: minimum and maximum values used for normalization.
- `heat_norm`: 0..1 scale used in visualizations and to derive relative threshold.
- `score`: maximum value of `heat` within the ROI (if it exists), otherwise global maximum.

**Interpretation:**
- Greater distances => more anomalous.
- Score > threshold => `is_anomaly = True`.

**Effective threshold:**
```python
threshold_base = THRESHOLD (env or config)
if mode == "sensitive": threshold = threshold_base * 0.8
elif mode == "strict":  threshold = threshold_base * 1.2
if thr (query param) != None: threshold = thr  (total override)
```

**Threshold normalization for mask:**
```python
thr_norm = (threshold - hmin) / (hmax - hmin + 1e-8)
```
It is used to segment the normalized map into normal and anomalous regions, facilitating visualization and binary mask generation.

---

## 9. Visualization and Polygons
This section introduces the generation of visual results that help interpret detected anomalies. It explains how heatmaps, binary masks, polygons delimiting anomalous zones are created, and how all this is saved as files accessible from the frontend.

**Process in `save_visuals_and_polys`:**
1. Convert `heat_norm` to 8 bits (0–255).
2. Generate colormap (JET).
3. Overlay colormap with the original grayscale image.
4. Binarize using `thr_norm` if provided; otherwise, apply 98th percentile of `values > 0` as adaptive threshold.
5. Apply morphological operations (*open* and *close*) to remove noise and close gaps.
6. Extract contours: filter by `AREA_MIN`.
7. Approximate polygons with `cv2.approxPolyDP` (fixed epsilon 2.0) to simplify geometry.
8. Draw polygons on overlay.
9. Add ROI border if it exists.
10. Save three files:
   - `*_overlay.png`
   - `*_heat.png`
   - `*_mask.png`
11. Build public URLs (`/static/overlays/...`).

**Logical control:**  
- Polygons are only returned if `is_anomaly = True`.  

**Example partial response:**
```json
{
  "score": 0.57,
  "threshold": 0.42,
  "is_anomaly": true,
  "polygons": [[[12,45],[38,44],[41,70],[10,72]]],
  "overlay_url": "/static/overlays/piece_overlay.png"
}
```

This way, the system not only calculates the anomaly but also offers a clear visual representation accessible from the frontend.

---

## 10. API Endpoints
This section documents the different endpoints offered by the backend. It explains what each one does, what parameters it accepts, what type of responses it returns, and how to interact with them from a browser, Python scripts, or via CURL.

### 10.1 GET /health
- Method: GET
- Body: none
- Response 200:
```javascript
{
  "status": "ok",
  "device": "cuda" | "cpu",
  "img_size": 256,
  "knn_k": 3,
  "threshold": 0.35,
  "ignore_border_pct": 0,
  "roi_path": null | "path"
}
```

### 10.2 GET /
- Method: GET
- Serves `templates/index.html` if it exists; if not:
```json
{ "detail": "templates/index.html not found" }
```
- Usage: deliver simple frontend (upload image, see overlay).

### 10.3 POST /predict
- Method: POST
- Content-Type: multipart/form-data
- Query Parameters (optional):
  - `thr`: float (manual threshold)
  - `mode`: "sensitive" | "strict"
    - `sensitive` => reduces threshold by 20%
    - `strict` => increases threshold by 20%
- Form Field:
  - `file`: image (jpeg/png)
- Responses:
  - **200 OK**:
    ```javascript
    {
      "score": float,
      "threshold": float,
      "is_anomaly": bool,
      "polygons": [ [ [x,y], ... ], ... ],
      "overlay_url": "/static/overlays/xxx_overlay.png" | null
    }
    ```
  - **400 Bad Request**:
    - "Empty file."
    - "Could not decode image."
  - **500 Startup Error** (if artifacts are missing):
    - "Memory bank does not exist: ..." (thrown during initial load)

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/predict?mode=strict" \
  -F "file=@./examples/piece123.png"
```

**Example (Python requests):**
```python
import requests
with open("piece123.png", "rb") as f:
    files = {"file": ("piece123.png", f, "image/png")}
    r = requests.post("http://localhost:8000/predict", files=files, params={"thr":0.4})
print(r.json())
```

---

## 11. Usage Examples in Different Scenarios
Here, practical examples are presented showing how to use the API in different contexts: quick tests, audits, or executions without visualization. These examples help better understand the real use of the endpoints and how to leverage their parameters.

1. **Flexible detection**  
   - Dynamically adjust the threshold for a batch of images with more noise: use `mode=sensitive`.
   - For greater rigor, use `mode=strict`, which increases the threshold by 20%.
     
2. **Audit** 
   - Call `/health` to verify that the loaded model version matches expectations (threshold, IMG_SIZE).
     
3. **Visualization disabled** 
   - Run with `SAVE_VIS=0` to reduce I/O if only JSON is required.
   - In this mode, no visual files (`overlay.png`, `heat.png`, `mask.png`) are generated, and the response is limited to JSON with score, threshold, and polygons.  

---

## 12. Integration with Frontend
This section explains how the frontend connects with the API. It shows how the main endpoint is used to upload images and visualize results, and proposes ideas to expand the interface (such as threshold sliders or mode selection).

- `GET /` delivers `index.html` which acts as a minimal reference frontend. This can include:
  - Form to upload image.
  - `fetch` call to the `/predict` endpoint.
  - Result rendering via `overlay_url`.
- Generated files (overlays, heatmaps, and masks) are stored in `/static/overlays/` and can be consumed directly by the frontend.
- Possible interface extensions:
  - Display polygons on an interactive canvas.
  - Slider to adjust threshold (`thr`) on the client, invoking `/predict?thr=...`.
  - Mode selector (`sensitive` or `strict`) to dynamically modify detector behavior.

This way, the frontend can offer an interactive and configurable experience for anomaly detection.

---

## 13. Dependencies (requirements.txt)
Here, the main libraries of the project are listed and the role of each is briefly explained. Recommendations are also given on compatibility and version pinning, especially for more sensitive components like PyTorch and CUDA.

**Main dependencies:**
- fastapi: web framework for building the API.
- uvicorn: ASGI server to run the application.
- python-dotenv: loading variables from `.env`.
- numpy, opencv-python (cv2): image processing and numerical operations.
- torch, torchvision: backbone (ResNet18) and tensor operations.
- scikit-learn: KNN implementation for the memory bank.
- *(Optional)* `pillow`: required by `torchvision` for some image transformations.
- *(Optional)* `jinja2`, `python-multipart`: useful if templates or forms are used in the frontend.

**Impact and recommendations:**
- Verify compatibility between `torch` and the installed CUDA version.  
- Pin versions in `requirements.txt` (example: `torch==2.0.1`, `fastapi==0.103.0`) to ensure memory bank reproducibility and stability in deployments.  
- Document optional dependencies according to the project's actual use.

Version pinning ensures the environment is reproducible and avoids incompatibilities in production.

---

## 14. Model Artifacts and Compatibility
### 14.1 Model Artifacts
Artifacts are the files that store the system's knowledge and are essential for the backend to start correctly.

**Default expected path:**
- `Backend/models/patchcore/memory_bank_core.npz`
- `Backend/models/patchcore/config.json`

You can change the path with the environment variable `ARTIFACTS_DIR` (e.g., `ARTIFACTS_DIR=Backend/models/patchcore`).

**Options to have the artifacts:**
- **A) Use pre-built artifacts**
  1. Obtain the files `memory_bank_core.npz` and `config.json` from your internal source (drive/release artifacts).
  2. Place them in `Backend/models/patchcore/` (or in the path configured in `ARTIFACTS_DIR`).
  3. Verify their existence:
     - Linux/macOS: `ls -lh Backend/models/patchcore/`
     - Windows (PowerShell): `Get-ChildItem Backend/models/patchcore/`
  > Note: If missing, the app will fail on startup with "Memory bank does not exist: …".

- **B) Generate artifacts from normal images**
  1. Gather a dataset of normal images (without defects), for example in `data/normal/`.
  2. Use the same preprocessing as inference (grayscale, `IMG_SIZE`).
  3. Extract embeddings with the backbone (ResNet18 with hooks on `layer2` and `layer3`), normalize per patch (L2), and build the KNN bank.
  4. Save:
     - `memory_bank_core.npz`: embeddings/memory matrix.
     - `config.json`: parameters (e.g., `{"threshold": 0.35, "embedding_version": "resnet18_layer2_3_concat_v1", "model_version": "1.0.0"}`).
  5. Example command (adjust to your script/notebook):
     ```javascript
     python tools/build_memory_bank.py --data-dir data/normal \
       --out-dir Backend/models/patchcore --img-size 256 --k 3 --threshold 0.35
     ```
  6. Later verification: start the backend and query `GET /health` to verify `threshold`, `img_size`, etc.

---

### 14.2 Compatibility (Python/CUDA)
To ensure reproducibility and performance, it is recommended:

- **Python version**: 3.10 or 3.11.  
- **PyTorch/torchvision**: install versions compatible with your environment (CPU or GPU).  
  - Official guide: [PyTorch Get Started](https://pytorch.org/get-started/locally/)  
- **GPU (optional)**:
  - Ensure the `torch` version matches your CUDA version (e.g., CUDA 12.1 ↔ build cu121).
  - Quick verification in Python:
    ```python
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
    ```
- **Version pinning recommendation (requirements.txt):**
  - Pin `torch`, `torchvision`, `fastapi`, `uvicorn`, `opencv-python`, etc., for reproducibility.
  - If using GPU, document which `torch` build to install (CPU-only vs cuXXX).

---

### 14.3 Test Execution with pytest
Tests allow validating the correct functioning of the backend.

**Test dependency installation:**
- `pip install pytest`
- (Optional) if your tests use HTTP clients: `pip install httpx pytest-asyncio`

**Prerequisites:**
- Ensure artifacts (`memory_bank_core.npz`, `config.json`) exist in the configured path.
- To speed up, you can disable visualizations in tests:
  - Linux/macOS: `export SAVE_VIS=0`
  - Windows (PowerShell): `$env:SAVE_VIS="0"`

**Execution:**
- Run all tests (for example, if they are in `Backend/tests`):
    - `pytest -q Backend/tests`
- Run a specific test:
    - `pytest -q Backend/tests/test_health.py -k test_health`
 
>**Notes:**
>  - If tests use FastAPI TestClient, you don't need to start `uvicorn`; tests import the app directly.
>  - If a test fails with artifact error, check the path (`ARTIFACTS_DIR`) or place the files in `Backend/models/patchcore/`.

---

## 15. Tests
This section explains the types of recommended tests to ensure the backend functions correctly. It includes examples of basic tests (health, prediction, expected errors) and suggestions for validating specific cases such as ROI usage or sensitivity modes.

**Suggested cases:**
- Test `/health` ⇒ 200 response and expected fields.  
- Test `/predict` with valid image ⇒ numeric `score` and `threshold`.  
- Test `/predict` with empty file ⇒ 400 response.  
- Test `/predict` with missing artifacts ⇒ 500 response (example: non-existent memory bank).  
- Test ROI ⇒ configure `IGNORE_BORDER_PCT` and verify score reduction outside the border.  
- Test `mode` ⇒ compare `is_anomaly` result with and without `mode=sensitive` given same score.  
- Test visualization ⇒ verify that `overlay_url` points to `/static/overlays/...`.

  
**Conceptual example (pytest):**
```python
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "device" in data
```

---

## 16. Security and Performance
This part gathers best practices to keep the backend stable, fast, and secure. Measures such as file validation, input size limitation, component caching, and proper CORS configuration in production environments are discussed.

**Recommendations:**
- **File size limit**: implement additional middleware to avoid excessive loads.  
- **Validate MIME types and extensions**: ensure only valid images (JPEG/PNG) are accepted.  
- **Cache backbone and KNN**: already performed at *startup* to reduce inference latency.  
- **Avoid arbitrary overlay overwriting**: sanitize base name and generate unique identifiers (UUID/timestamp).  
- **Horizontal scaling**: use load balancer and replicate *memory bank* in *read-only* mode.  
- **High data volumes**: consider batching (not implemented) or reduce `IMG_SIZE` to optimize performance.  
- **CORS**: currently open (`allow_origins=["*"]`); in production restrict to trusted domains.

Applying these measures ensures a robust, efficient, and secure backend in production environments.

---

## 17. Future Extensions
Here, ideas and improvement lines that could be implemented in the future are presented, such as batch processing, authentication, database persistence, or exposure of performance metrics.

**Ideas:**
- **Endpoint `GET /config`**: expose additional metadata (memory bank version, active parameters like `IMG_SIZE`, `knn_k`, `threshold`).  
- **Endpoint `POST /predict/batch`**: allow sending multiple images in a single request and return parallel results.  
- **Authentication**: add token support (JWT, OAuth2) to restrict access to sensitive endpoints.  
- **Prometheus Metrics**: export latency indicators, inference counts, errors, and resource usage (CPU/GPU).  
- **Result persistence**: store inferences in a lightweight database (SQLite for development, PostgreSQL for production).  
- **WebSocket**: enable real-time notifications to show progress in heavy tasks (preprocessing or batch).  

These extensions would allow scaling the system, improving security, and offering greater observability in production environments.

---

## 18. ASCII Architecture Diagrams
This section shows text diagrams that help visualize the relationship between the different components of the system. They are useful for understanding at a glance how information flows from the frontend to inference and response.

### 18.1 Components
``` draw.io
+-------------------+          +--------------------------+
|  Client (Web)     |  HTTP    | FastAPI (/predict,/...)  |
|  - index.html     | <------> | main.py                  |
|  - JS fetch       |          |                          |
+-------------------+          +-----------+--------------+
                                           |
                                           | Startup
                                           v
                               +----------------------------+
                               |  Backbone (ResNet18)       |
                               |  Hooks layer2 / layer3     |
                               +-------------+--------------+
                                             |
                                +------------v-------------+
                                |  Memory Bank (KNN)       |
                                |  (normal embeddings)     |
                                +------------+-------------+
                                             |
                                 +------------v-------------+
                                |  Inference                |
                                |  - Patchify               |
                                |  - KNN Distances          |
                                |  - Heat / Score / ROI     |
                                |  - Threshold comparison   |
                                +------------+--------------+
                                             |
                                +------------v-------------+
                                |  Visualizations          |
                                |  overlays / polygons     |
                                |  files in /static/...    |
                                +------------+-------------+
                                             |
                                +------------v-------------+
                                |  JSON Response           |
                                |  score, threshold,       |
                                |  is_anomaly, polygons,   |
                                |  overlay_url             |
                                +--------------------------+
```

---

## 20. Complete Inference Cycle Example
Here, a complete practical case, step by step, of how the backend processes a real image is shown. It allows concretely understanding how parameters are applied and how the final response is interpreted.

Given:
- Image `piece123.png`
- `.env` with `THRESHOLD=0.40`
- Call: `POST /predict?mode=strict`

**Flow:**
1. Base threshold = 0.40 → `strict` mode increases by 20% ⇒ effective threshold = 0.48.  
2. `score = 0.52` is calculated.  
3. Since `score > threshold`, the image is determined to be an anomaly.  
4. Overlays are generated and, after binarization and morphological operations, two contours are detected.  
5. JSON Response:
```json
{
  "score": 0.52,
  "threshold": 0.48,
  "is_anomaly": true,
  "polygons": [
    [[15,34],[44,33],[47,60],[13,62]],
    [[120,88],[150,87],[151,110],[119,112]]
  ],
  "overlay_url": "/static/overlays/piece123_overlay.png"
}
```
*(Polygons are returned only if `is_anomaly = true`.)*

This example shows how configuration parameters and the backend's internal flow are directly reflected in the final response consumed by the frontend.

---

## 21. Final Summary
This last part condenses the main points of the entire documentation. It summarizes the purpose of the backend, its flexibility, visualization capability, and possible expansion paths for future projects.

**The backend:**
- **Core**: offers efficient anomaly inference with PatchCore (ResNet18 + KNN).  
- **Configuration and visualization**: configurable via environment (`.env`), provides optional visualizations for human analysis, and allows sensitivity adjustment via `mode` or `thr` parameter.  
- **Integration and extensibility**: facilitates integration with a basic frontend and is extensible towards batching, authentication, and persistent storage.  

This backend constitutes a solid foundation for anomaly detection projects and can evolve towards more complex and scalable solutions.

---
