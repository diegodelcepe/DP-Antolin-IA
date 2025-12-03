
---

# PatchCore Anomaly Inspector

### *FastAPI • Docker GPU • ResNet18 • PatchCore KNN •* 

Este proyecto implementa un sistema de **detección de anomalías** basado en **PathCore**, usando extracción de características con **ResNet-18** y un **memory bank precomputado**
Incluye:

* API de inferencia basada en **FastAPI**
* Ejecución acelerada por **GPU (CUDA)** mediante Docker.
* Web UI en **HTML + JS**, con:

  * Análisis por lotes.
  * Vista de galería y tabla.
  * Métricas (IoU, áreas de defecto, score).
  * Modo **cámara en tiempo real** con overlays.
* Logging completo de predicciones (`predictions.csv`).
* Soporte de **ROI** (overlays, heatmaps, máscaras y logging estructurado).

Está diseñado para integrarse en pipelines industriales o pruebas de concepto de inspección de calidad.

---

# Estructura del proyecto

```
docker-mvp/
├─ Backend/
│  ├─ models/patchcore/
│  │   ├─ config.json
│  │   └─ memory_bank_core.npz
│  ├─ static/
│  │   ├─ assets/               # Frontend (CSS + JS)
│  │   └─ overlays/             # Overlays generados 
│  ├─ templates/
│  │   └─ index.html            # Interfaz web
│  ├─ tests/
│  ├─ .env                      # Configuración del backend
│  ├─ main.py                   # FastAPI (núcleo PatchCore)
│  └─ requirements.txt
│
├─ Dataset/                     # dataset local
├─ notebooks/                   
├─ Dockerfile.gpu
├─ docker-compose.yml
└─ README.md
```

> El directorio /Backend/logs/ se genera automáticamente y contiene predictions.csv

---

# Arquitectura del Backend

### 1. Preprocesamiento

* Redimensionado a IMG_SIZE × IMG_SIZE (por defecto: 256×256).
* Transformación a escala de grises.
* Replicado a 3 canales (GRAY → BGR).
* Conversión a BGR.

### 2. Extracción de características (ResNet-18)

* Hooks en **layer2** y **layer3**.
* layer3 se interpola para que coincida con las dimensiones de layer2.
* Ambas se concatenan a lo largo de canales.

Resultado: mapa de características combinado.

### 3. PatchCore + KNN

* Patchificación del feature map.
* Normalización L2 fila a fila (estabilidad numérica).
* Búsqueda de **K vecinos más cercanos** usando sklearn.neighbors.NearestNeighbors
* Cálculo del heatmap mediante distancia promedio al memory bank:

  ```
  Backend/models/patchcore/memory_bank_core.npz
  ```

### 4. Score de anomalía

* Score = valor máximo del heatmap:

  * Si existe ROI → máximo DENTRO de la ROI.
  * Si no hay ROI → máximo global.

### 5. Visualizaciones

Si `SAVE_VIS=1`, se generan en /Backend/static/overlays:

* Overlay a color (`*_overlay.png`)
* Heatmap (`*_heat.png`)
* Máscara binaria de defectos (`*_mask.png`)+

### 6. Polígonos y áreas

* Detección de contornos con OpenCV.
* Áreas individuales y total.
* Polígonos aproximados.

---

# ROI (Región de interés)

El sistema permite recortar la evaluación mediante dos mecanismos:

### ROI por porcentaje de borde: `IGNORE_BORDER_PCT`

Ignora un borde proporcional. Ejemplo: si vale `5`, ignora 5% del borde en cada lado.

### ROI por máscara externa: `ROI_PATH`

Archivo PNG binario donde:

* 255 → píxel válido
* 0 → ignorar

Se usa para:

* Score
* Binarización del heatmap
* Polígonos
* IoU aproximado

---

# Métrica IoU

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

# Logging automático

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
* IoU real / aproximado
* overlays URLs

---

# Interfaz Web (Frontend)

La interfaz web (`templates/index.html` + `static/assets/app.js`) incluye:

### Carga de imágenes

* Múltiples archivos
* Carpetas completas (`webkitdirectory`)
* Drag & Drop

### Configuración de umbral

* Automático (backend)
* Manual (usuario)
* Modos de sensibilidad:

  * **normal**
  * **sensitive (0.8×)**
  * **strict (1.2×)**

### Resultados

* Vista **galería** (cards)
* Vista **tabla**
* Exportación **CSV**
* Panel de **KPIs**:
  Total, normales, anomalías, tasa de defectos, área media.

### Modo cámara en tiempo real

Captura frames cada 1.5s y los envía a `/predict?source=camera`.
Muestra:

* Overlay
* Score
* Área total de defecto
* Estado
* IoU aproximado

---

# Variables de Entorno (Backend/.env)

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

# Ejecución con Docker

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

# Construir y ejecutar

### Primera vez o después de cambios en el backend

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

### ``GET /`

Devuelve la interfaz web.

### `GET /health`

Configuración básica del backend.

```json
{
  "status": "ok",
  "device": "cuda",
  "img_size": 256,
  "knn_k": 3,
  "threshold": 0.35
}
```

### `POST /predict`

Predicción de una sola imagen

Query params:

* `thr`
* `mode`
* `source=camera|upload|single`

---

### `POST /predict_batch`

Procesa múltiples imágenes y devuelve:

* summary (KPIs)
* result por imagen
* overlays
* áreas
* IoU aproximado

---

# Notas finales

* Si hay GPU, el modelo usa **CUDA** automáticamente.
* El sistema está preparado para producción y uso industrial.
* Web UI pensada para operadores, laboratorio y validación rápida.
* Backend modular y fácil de extender mediante hooks.

---

