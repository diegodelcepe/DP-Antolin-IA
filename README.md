
---

# 📦 PatchCore Anomaly Inspector

### *FastAPI • Docker GPU • ResNet18 • KNN PatchCore • Web UI en tiempo real*

Este sistema implementa un **inspector automático de defectos** basado en el método **PatchCore** (KNN sobre *memory bank*) con extracción de características usando **ResNet-18**.
Incluye:

* API de inferencia en **FastAPI** (optimizada para GPU, Docker & CUDA).
* Web UI moderna en **HTML + JS**, con:

  * Análisis por lotes.
  * Vista de galería y tabla.
  * Métricas (IoU, áreas de defecto, score).
  * Modo **cámara en tiempo real** con overlays.
* Logging completo de predicciones (`predictions.csv`).
* Soporte avanzado de ROI (máscaras externas + porcentaje de borde).
* Generación de overlays, heatmaps y máscaras de defectos.

Este repositorio está listo para uso en entornos industriales, POCs y pipelines de inspección de calidad.

---

# 📁 Estructura del proyecto

```
docker-mvp/
├─ Backend/
│  ├─ models/patchcore/
│  │   ├─ config.json
│  │   └─ memory_bank_core.npz
│  ├─ static/
│  │   ├─ assets/               # Frontend (CSS + JS)
│  │   └─ overlays/             # Overlays generados (montado como volumen)
│  ├─ templates/
│  │   └─ index.html            # Interfaz web completa
│  ├─ tests/
│  ├─ .env                      # Configuración del backend
│  ├─ main.py                   # FastAPI (núcleo PatchCore)
│  └─ requirements.txt
│
├─ Dataset/                     # (opcional) dataset local
├─ notebooks/                   
├─ Dockerfile.gpu
├─ docker-compose.yml
└─ README.md
```

> La carpeta `logs/` no necesita existir:
> el backend la crea automáticamente (`LOGS_DIR`) y genera `predictions.csv`.

---

# ⚙️ Arquitectura del Backend

## 🔧 Flujo general del modelo

### 1. Preprocesamiento

* Redimensionado a `IMG_SIZE × IMG_SIZE` (por defecto: 256×256).
* Conversión a escala de grises.
* Replicado a 3 canales para ResNet-18.

### 2. Extracción de características

* Backbone **ResNet-18** preentrenada.
* Hooks en `layer2` y `layer3`.
* Concat: `layer2 || upsample(layer3)` → mapa combinado de features.

### 3. PatchCore + KNN

* Patchify del feature map.
* Normalización L2 fila a fila (estabilidad numérica).
* Memory bank cargado desde:

  ```
  Backend/models/patchcore/memory_bank_core.npz
  ```
* KNN (`KNN_K` vecinos).
* Mapa de calor = distancia promedio a K vecinos.

### 4. Score global

* Reescalado del mapa de calor a 256×256.
* Score = valor máximo del mapa:

  * Si hay ROI → máximo DENTRO de la ROI.
  * Si no hay ROI → máximo global.

### 5. Visualizaciones

Si `SAVE_VIS=1`, se generan:

* Overlay (`*_overlay.png`)
* Heatmap (`*_heat.png`)
* Máscara (`*_mask.png`)

### 6. Polígonos y áreas

* Detección de contornos > `AREA_MIN`.
* Se devuelven polígonos, áreas individuales y totales.

---

# 🎯 ROI (Región de interés)

Sistema flexible de recorte lógico aplicado al mapa de calor:

### ✔️ ROI por porcentaje: `IGNORE_BORDER_PCT`

Ejemplo: si vale `5`, ignora 5% del borde en cada lado.

### ✔️ ROI por máscara externa: `ROI_PATH`

PNG binaria donde:

* 255 → píxel válido
* 0 → ignorar

Se usa para:

* Score
* Binarización del heatmap
* IoU aproximado

---

# 📊 Métrica IoU

### IoU real (si se envía `gt_mask`)

En `/predict`:

```
iou = intersection(pred_mask, gt_mask) / union(pred_mask, gt_mask)
```

### IoU aproximado (si no hay máscara GT)

```
IoU_approx = área_total_defectos / área_ROI
```

Siempre normalizado a `[0, 1]`.

---

# 🧾 Logging automático (predictions.csv)

Cada predicción se guarda en:

```
Backend/logs/predictions.csv
```

Incluye:

* timestamp
* source (single, batch, camera…)
* filename
* score
* threshold
* is_anomaly
* defect_area_total_px
* defect_area_max_px
* iou

---

# 🌐 Frontend Web (UI)

La interfaz web (`templates/index.html` + `static/assets/app.js`) incluye:

### ✔️ Carga de imágenes

* Múltiples archivos
* Carpetas completas (`webkitdirectory`)
* Drag & Drop

### ✔️ Configuración de umbral

* Automático (backend)
* Manual (usuario)
* Modos de sensibilidad:

  * **normal**
  * **sensitive (0.8×)**
  * **strict (1.2×)**

### ✔️ Resultados

* Vista **galería** (cards)
* Vista **tabla**
* Exportación **CSV**
* Panel de **KPIs**:
  Total, normales, anomalías, tasa de defectos, área media.

### ✔️ Modo cámara en tiempo real

Captura frames cada 1.5s y los envía a `/predict?source=camera`.
Muestra:

* Overlay
* Score
* Área total de defecto
* Estado
* IoU aproximado

---

# 🔧 Variables de Entorno (Backend/.env)

Ejemplo:

```env
# Artefactos PatchCore
ARTIFACTS_DIR=models/patchcore

# Estáticos
STATIC_DIR=static
OVERLAYS_SUBDIR=overlays

# Logs
LOGS_DIR=logs

# Parámetros del modelo
THRESHOLD=0.35
IMG_SIZE=256
KNN_K=3
PATCH_STRIDE=1
SAVE_VIS=1
AREA_MIN=200

# ROI
IGNORE_BORDER_PCT=0
ROI_PATH=
```

---

# 🐳 Docker (GPU)

## Imagen: `Dockerfile.gpu`

* Base: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
* Python 3.11 + pip
* Install requirements
* Copia todo el repo en `/app`
* Crea el directorio de overlays
* Ejecuta uvicorn:

```bash
python -m uvicorn Backend.main:app --host 0.0.0.0 --port 8000
```

## docker-compose.yml (servicio `api`)

Volúmenes clave:

```yaml
- ./Backend/static/overlays:/app/Backend/static/overlays
- ./Backend/models/patchcore:/app/Backend/models/patchcore:ro
- ./Backend/templates:/app/Backend/templates:ro
- ./Backend/static/assets:/app/Backend/static/assets:ro
- ./Dataset:/app/Dataset:ro
```

GPU habilitada mediante:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - capabilities: ["gpu"]
```

---

# 🚀 Cómo correr el sistema

### 👉 Primera vez o después de cambios en el backend

```bash
docker compose up -d --build
```

o en dos pasos:

```bash
docker compose build api
docker compose up -d api
```

Si quieres reconstruir sin cache:

```bash
docker compose build --no-cache api
docker compose up -d api
```

### Ver logs en tiempo real

```bash
docker compose logs -f api
```

### Detener el servicio

```bash
docker compose down
```

---

# 🔌 Endpoints de la API

### `GET /health`

Config actual del modelo.

```json
{
  "status": "ok",
  "device": "cuda",
  "img_size": 256,
  "knn_k": 3,
  "threshold": 0.35
}
```

---

### `GET /`

Sirve la interfaz web (UI).

---

### `POST /predict`

📥 Una sola imagen (opcional `gt_mask` para IoU real).

Query params:

* `thr`
* `mode`
* `source=camera|upload|single`

---

### `POST /predict_batch`

📥 Múltiples imágenes.

Devuelve:

* summary (KPIs)
* result por imagen
* overlays
* áreas
* IoU aproximado

---

# 🧠 Notas finales

* Si hay GPU, el modelo corre en **CUDA** automáticamente.
* El sistema está preparado para producción y uso industrial.
* Web UI pensada para operadores, laboratorio y validación rápida.
* Backend modular y fácil de extender mediante hooks.

---

