# syntax=docker/dockerfile:1
FROM python:3.10-slim

# Ajustes básicos de Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Paquetes del sistema que OpenCV/Torch suelen necesitar
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git libgl1 libglib2.0-0 ffmpeg libsm6 libxext6 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Fijar la caché
ENV TORCH_HOME=/app/.cache/torch


# 1) Instalar dependencias (se cachea mejor separando el COPY)
COPY Backend/requirements.txt ./Backend/requirements.txt
RUN pip install --upgrade pip && pip install -r Backend/requirements.txt

# 2) Copiar el código (incluye artefactos del modelo si están en esa carpeta)
COPY Backend ./Backend

# Carpeta donde se guardan los overlays de resultados
RUN mkdir -p Backend/static/overlays

# Exponer el puerto
EXPOSE 8000

# precarga el peso de ResNet18 en build (se queda como capa de la imagen)
RUN python - <<'PY'
import torch
from torchvision.models import resnet18, ResNet18_Weights

torch.hub.set_dir("/app/.cache/torch")
# fuerza la descarga en build
resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print("ResNet18 weights baked into the image.")
PY



# Arrancar la API (FastAPI con Uvicorn)
CMD ["uvicorn", "Backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
