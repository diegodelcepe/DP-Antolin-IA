
---

# PatchCore Anomaly Inspector

## FastAPI • Docker GPU • ResNet18 • PatchCore KNN

Sistema de detección de anomalías industrial basado en PatchCore (ResNet-18 + Memory Bank). Diseñado para inspección de calidad automatizada con soporte de GPU.

## Características Principales

* Core: Extracción de características con ResNet-18 y detección por vecindad (KNN).

* Interfaz Web: Dashboard para análisis por lotes, métricas (IoU, Score), heatmaps y modo cámara en tiempo real.

* API: Backend robusto en FastAPI.

* ROI Avanzado: Soporte para máscaras de exclusión y márgenes de borde.

* Logging: Registro automático de predicciones en CSV.

## Puesta en Marcha (Docker)
```
El proyecto está contenerizado y configurado para usar NVIDIA GPU automáticamente.

# 1. Construir y levantar el servicio
docker compose up -d --build

# 2. Ver logs en tiempo real
docker compose logs -f api

# 3. Detener
docker compose down
```

Acceso: La interfaz web estará disponible en http://localhost:8000.

## Estructura Clave
```
docker-mvp/
├─ Backend/
│  ├─ models/patchcore/   # Memory bank y config (.npz, .json)
│  ├─ static/overlays/    # Salida de heatmaps y máscaras generadas
│  ├─ logs/               # Contiene predictions.csv
│  ├─ .env                # Configuración principal
│  └─ main.py             # Entrypoint FastAPI
├─ Dockerfile.gpu
└─ docker-compose.yml
```

## Configuración (.env)

Los parámetros del modelo se ajustan en Backend/.env. Las variables más importantes:

* THRESHOLD: Umbral de decisión (def: 0.35).

* IMG_SIZE: Resolución de entrada (def: 256).

* SAVE_VIS: 1 para guardar imágenes de debug (overlays/heatmaps).

* ROI_PATH: Ruta a la máscara PNG para limitar la zona de inspección.

## API Endpoints

| Método | Endpoint | Descripción |
| :--- | :--- | :--- |
| `GET` | `/` | Carga la Interfaz Web (UI). |
| `GET` | `/health` | Estado del servicio y config actual. |
| `POST` | `/predict` | Inferencia de imagen única (params: `thr`, `source`). |
| `POST` | `/predict_batch` | Procesa múltiples imágenes y devuelve métricas. |

## Arquitectura Técnica

1. Preproceso: Redimensión y normalización.

2. Extracción: Hooks en layer2 y layer3 de ResNet-18.

3. PatchCore: Comparación con memory bank usando KNN ($K=3$).

4. Scoring: Máximo valor del mapa de calor dentro de la ROI válida.

5. Métricas: Cálculo de IoU (Real si hay GT, aproximado si no).

---
