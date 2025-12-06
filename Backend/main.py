"""
PatchCore Anomaly API (FastAPI)
--------------------------------
Servicio de inferencia para inspección de piezas con el método tipo PatchCore + KNN.

Características principales:
- Carga de configuración desde variables de entorno y/o `models/patchcore/config.json`.
- Preprocesamiento a escala fija (IMG_SIZE) y conversión a tensor 3 canales (gris replicado).
- Extracción de características desde ResNet-18 (layers 2 y 3), con *hooks*.
- Normalización L2 por parches y búsqueda de vecinos más cercanos (KNN) en un *memory bank*.
- Mapa de calor de anormalidad (distancia promedio a K vecinos) y puntuación global.
- Soporte a ROI: ignorar bordes por porcentaje y/o usar una máscara binaria externa.
- Generación de *overlays* (mapa de calor + contornos por umbral) y guardado en `static/overlays`.
- Endpoints:
    - `/health`: info básica del servicio.
    - `/`: sirve un HTML simple si existe `templates/index.html`.
    - `/predict`: una sola imagen, con opción de máscara ground truth para IoU.
    - `/predict_batch`: varias imágenes en lote, con resumen de métricas.
- Registro automático de predicciones (score, estado, áreas, IoU) en CSV.

Métrica IoU:
- Si se proporciona `gt_mask`, se calcula IoU “real” predicción vs ground truth.
- Si NO se proporciona `gt_mask`, se usa un IoU aproximado:
    IoU ≈ área_defecto_total / área_ROI

NOTA: Esta versión añade documentación, métricas extra (áreas, IoU) y logging,
      y mantiene el cálculo de `threshold` compatible con el frontend
      (cada resultado de batch incluye su propio `threshold`).
"""

# =======================
# Imports
# =======================
import os
import json
import math
import csv
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import cv2
import torch
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi import Form

from dotenv import load_dotenv  # Carga variables desde .env si existe

# =======================
# Paths y helpers
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _abs(path: str) -> str:
    """
    Devuelve una ruta absoluta.

    - Si `path` es absoluta, se devuelve tal cual.
    - Si `path` es relativa, se resuelve desde `BASE_DIR`.
    """
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


# Cargar .env (si existe) ANTES de leer os.getenv(...)
# `override=False` evita sobreescribir variables ya presentes en el entorno del proceso
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"), override=False)

# =======================
# Config por entorno
# =======================
# Ruta base de artefactos del modelo (memory bank, config.json, etc.).
ARTIFACTS_DIR = _abs(os.getenv("ARTIFACTS_DIR", os.path.join("models", "patchcore")))
# Directorio para servir archivos estáticos (overlays, etc.).
STATIC_DIR = _abs(os.getenv("STATIC_DIR", "static"))
# Subcarpeta donde se guardarán las imágenes generadas (overlays, heatmaps, masks).
OVERLAYS_SUBDIR = os.getenv("OVERLAYS_SUBDIR", "overlays")

# Directorio y archivo para logs de predicciones
LOGS_DIR = _abs(os.getenv("LOGS_DIR", "logs"))
os.makedirs(LOGS_DIR, exist_ok=True)
PREDICTION_LOG_PATH = os.path.join(LOGS_DIR, "predictions.csv")

# Cargar JSON de configuración si existe (permite fijar threshold, etc.).
CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "config.json")
CONFIG_JSON: dict = {}
if os.path.exists(CONFIG_PATH):
    try:
        CONFIG_JSON = json.load(open(CONFIG_PATH, "r", encoding="utf-8"))
    except Exception:
        # Silencioso a propósito: si falla la lectura, continuamos con valores por defecto
        CONFIG_JSON = {}

# Hiperparámetros / ajustes
THRESHOLD = float(os.getenv("THRESHOLD", str(CONFIG_JSON.get("threshold", 0.35))))
IMG_SIZE = int(os.getenv("IMG_SIZE", "256"))        # Debe coincidir con el memory bank
KNN_K = int(os.getenv("KNN_K", "3"))
PATCH_STRIDE = int(os.getenv("PATCH_STRIDE", "1"))  # Submuestreo del mapa de features para KNN
SAVE_VIS = os.getenv("SAVE_VIS", "1") == "1"        # Guardar visualizaciones (overlay/heat/mask)

# Visual / polígonos
AREA_MIN = int(os.getenv("AREA_MIN", "200"))        # Área mínima (px) para considerar un contorno

# ROI: dos mecanismos (acumulables)
# 1) IGNORE_BORDER_PCT: ignora un porcentaje desde cada borde (en la imagen final IMG_SIZE×IMG_SIZE)
# 2) ROI_PATH: ruta a una máscara binaria (PNG) donde blanco=ROI válido y negro=ignorar
IGNORE_BORDER_PCT = float(os.getenv("IGNORE_BORDER_PCT", "0"))
ROI_PATH_ENV = os.getenv("ROI_PATH", "")
ROI_PATH = _abs(ROI_PATH_ENV) if ROI_PATH_ENV else ""

# Selección de dispositivo para PyTorch
device_available = torch.cuda.is_available()
DEVICE = "cuda" if device_available else "cpu"

# =======================
# FastAPI (app + CORS + estáticos)
# =======================
app = FastAPI(title="PatchCore Anomaly API", version="1.3.0")

# CORS abierto (útil para prototipos / pruebas). En producción, restringir dominios.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restringir dominios en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Asegurar que el directorio de estáticos exista y montarlo en /static
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# =======================
# Utilidades de imagen y tensores
# =======================


def imread_from_upload(file: UploadFile) -> np.ndarray:
    """
    Lee bytes de un UploadFile de FastAPI y devuelve BGR uint8 (OpenCV).

    Reglas:
    - Convierte GRAY→BGR y BGRA→BGR para uniformidad.
    - Lanza HTTP 400 si el archivo está vacío o si OpenCV no puede decodificar.
    """
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Archivo vacío.")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")
    if img.ndim == 2:  # Gris → BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # BGRA → BGR (descarta canal alpha)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def bgr_to_gray_256(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """
    Convierte BGR→GRIS y redimensiona a `size×size` con INTER_AREA.

    Devuelve:
        np.uint8 [0..255] con forma (size, size).
    """
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return g


def to_tensor_3ch(gray: np.ndarray) -> torch.Tensor:
    """
    Convierte una imagen en escala de grises a tensor [1,3,H,W] en DEVICE.

    Pasos:
    - Normaliza a [0,1].
    - Replica el canal gris a 3 canales.
    - Devuelve tensor en el dispositivo adecuado (CPU o GPU).
    """
    x = gray.astype(np.float32) / 255.0
    x = np.stack([x, x, x], axis=0)  # 3 canales idénticos (ResNet espera 3 canales)
    return torch.from_numpy(x).unsqueeze(0).to(DEVICE)


# =======================
# Backbone + Hooks (extracción de features)
# =======================
class FeatHook:
    """
    Pequeño contenedor para registrar un *forward hook* y almacenar la salida intermedia.

    Uso:
        layers = dict(backbone.named_modules())
        h2 = FeatHook(layers["layer2"])
        ...
        forward(x)
        features = h2.feat
    """

    def __init__(self, m):
        self.h = m.register_forward_hook(self._hook)
        self.feat = None

    def _hook(self, m, inp, out):
        # Guardar la activación (sin gradiente)
        self.feat = out.detach()

    def close(self):
        """Libera el hook cuando no se necesite."""
        self.h.remove()


def build_backbone() -> Tuple[torch.nn.Module, FeatHook, FeatHook]:
    """
    Construye ResNet-18 pre-entrenada y registra hooks en `layer2` y `layer3`.

    Returns:
        (backbone, hook_layer2, hook_layer3)
    """
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE)
    backbone.eval()
    layers = dict(backbone.named_modules())
    h2 = FeatHook(layers["layer2"])  # tamaño espacial más alto que layer3
    h3 = FeatHook(layers["layer3"])  # características más semánticas
    return backbone, h2, h3


def extract_concat_features(
    x: torch.Tensor,
    backbone: torch.nn.Module,
    h2: FeatHook,
    h3: FeatHook,
) -> torch.Tensor:
    """
    Realiza forward para poblar hooks y concatena features (layer2 || upsample(layer3)).

    Args:
        x: tensor [1,3,H,W]
        backbone: modelo ResNet-18
        h2, h3: hooks en layer2 y layer3

    Returns:
        Tensor [C_concat, Hf, Wf] con features concatenadas.
    """
    with torch.no_grad():
        _ = backbone(x)
    f2 = h2.feat  # [B,C2,H2,W2]
    f3 = h3.feat  # [B,C3,H3,W3]
    # Ajustar spatial de layer3 al de layer2
    f3u = torch.nn.functional.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    fcat = torch.cat([f2, f3u], dim=1).squeeze(0)  # [C2+C3, Hf, Wf]
    return fcat


def patchify_feature_map(fmap: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """
    Convierte un mapa de features [C,H,W] en una matriz de parches [N_patches, C].

    - Si `stride>1`, submuestrea espacialmente H y W para acelerar KNN
      a costa de resolución espacial.
    """
    C, H, W = fmap.shape
    if stride <= 1:
        return fmap.permute(1, 2, 0).reshape(H * W, C).contiguous()
    f = fmap[:, ::stride, ::stride]
    h, w = f.shape[-2:]
    return f.permute(1, 2, 0).reshape(h * w, C).contiguous()


def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    """Normalización L2 fila a fila con pequeña epsilon para evitar división por cero."""
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / n


# =======================
# ROI helpers
# =======================
ROI_MASK: Optional[np.ndarray] = None  # Máscara binaria IMG_SIZE×IMG_SIZE (255=ROI, 0=ignorar)


def build_roi_mask(img_size: int) -> Optional[np.ndarray]:
    """
    Crea una máscara ROI combinando dos fuentes:
    1) Borde ignorado por porcentaje (cuadro interior válido).
    2) Máscara binaria externa opcional (PNG).

    Returns:
        np.ndarray uint8 con valores 0/255, o None si la máscara final es toda cero.
    """
    mask = np.ones((img_size, img_size), np.uint8) * 255

    # 1) Ignorar bordes por porcentaje
    if IGNORE_BORDER_PCT > 0:
        m = int(round(img_size * IGNORE_BORDER_PCT / 100.0))
        if m > 0:
            mask[:m, :] = 0
            mask[-m:, :] = 0
            mask[:, :m] = 0
            mask[:, -m:] = 0

    # 2) Máscara desde archivo (si existe)
    if ROI_PATH and os.path.exists(ROI_PATH):
        m2 = cv2.imread(ROI_PATH, cv2.IMREAD_GRAYSCALE)
        if m2 is not None:
            m2 = cv2.resize(m2, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
            _, m2b = cv2.threshold(m2, 127, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_and(mask, m2b)

    if mask.max() == 0:
        # No hay píxeles válidos en la máscara resultante
        return None
    return mask


def get_roi_area_pixels() -> int:
    """
    Devuelve el número de píxeles válidos en la ROI.

    - Si hay ROI_MASK: cuenta píxeles >0.
    - Si no hay ROI: usa toda la imagen (IMG_SIZE x IMG_SIZE).
    """
    global ROI_MASK
    if ROI_MASK is not None:
        return int((ROI_MASK > 0).sum())
    return IMG_SIZE * IMG_SIZE


# =======================
# Scoring + Visuales
# =======================


def anomaly_map_and_score(
    gray_img: np.ndarray,
    backbone: torch.nn.Module,
    h2: FeatHook,
    h3: FeatHook,
    knn: NearestNeighbors,
    stride: int = PATCH_STRIDE,
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Calcula mapa de anormalidad y score global.

    Pasos:
    - Convierte la imagen gris a tensor 3ch y extrae features concatenadas (layer2+layer3↑).
    - *Patchify* y normaliza L2 cada vector de parche.
    - KNN: para cada parche, distancia promedio a sus K vecinos más cercanos del memory bank.
    - Reescala a IMG_SIZE y normaliza [0..1] (guardando hmin/hmax).
    - `score` = máx(heat) dentro de ROI si existe, si no, máx global de heat.

    Returns:
        heat        : mapa de calor crudo (float32).
        heat_norm   : mapa [0..1].
        hmin, hmax  : min y max del mapa crudo.
        score       : valor escalar en la escala cruda (heat).
    """
    # 1) features
    x = to_tensor_3ch(gray_img)
    fcat = extract_concat_features(x, backbone, h2, h3)
    Hf, Wf = fcat.shape[-2:]

    # 2) parches + normalización L2
    patches = patchify_feature_map(fcat, stride=stride)
    patches = torch.nn.functional.normalize(patches, p=2, dim=1).cpu().numpy()

    # 3) Distancias KNN (promedio sobre K) → mapa [Hf',Wf']
    dists, _ = knn.kneighbors(patches, return_distance=True)
    ph = dists.mean(axis=1).reshape(
        Hf if stride <= 1 else math.ceil(Hf / stride),
        Wf if stride <= 1 else math.ceil(Wf / stride),
    ).astype(np.float32)

    # 4) Reescalar al tamaño de la imagen final (IMG_SIZE×IMG_SIZE)
    heat = cv2.resize(ph, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    hmin, hmax = float(heat.min()), float(heat.max())
    heat_norm = (heat - hmin) / (hmax - hmin + 1e-8)

    # 5) Score (máx) restringido a ROI si existe
    if roi_mask is not None:
        heat_for_score = heat.copy()
        # Penaliza fuera de ROI para que nunca sea el máximo
        heat_for_score[roi_mask == 0] = hmin - 1.0
        score = float(heat_for_score.max())
    else:
        score = float(heat.max())

    return heat, heat_norm, hmin, hmax, score


def save_visuals_and_polys(
    img_gray: np.ndarray,
    heat_norm: np.ndarray,
    area_min: int,
    base_name: str,
    thr_norm: Optional[float] = None,
    roi_mask: Optional[np.ndarray] = None,
) -> Tuple[str, str, str, List[Any], List[float], str, str, str, float, float, np.ndarray]:
    """
    Genera y guarda visualizaciones y polígonos de defectos.

    Entradas:
      - `heat_norm`: mapa [0..1] (normalizado con hmin/hmax del caso).
      - `thr_norm`: umbral en [0..1] para binarizar (si None, usa percentil 98 de ROI>0).
      - `roi_mask`: si se provee, sólo se binariza dentro de ROI (para evitar falsos contornos fuera).

    Salidas:
      - Rutas absolutas a overlay, heat y mask.
      - Lista de polígonos (cada uno es lista de [x,y]).
      - Lista de áreas en píxeles por polígono.
      - URLs públicas `/static/...` equivalentes a los archivos.
      - Áreas total y máxima de defecto (en píxeles de la máscara binaria).
      - Máscara binaria usada (np.ndarray).
    """
    overlays_dir = os.path.join(STATIC_DIR, OVERLAYS_SUBDIR)
    os.makedirs(overlays_dir, exist_ok=True)

    # Preparar imágenes base
    raw_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    heat_u8 = (heat_norm * 255).astype(np.uint8)

    # Aplica ROI sólo para la binarización (no para el colormap)
    heat_u8_for_bin = cv2.bitwise_and(heat_u8, heat_u8, mask=roi_mask) if roi_mask is not None else heat_u8

    # Colormap y overlay semitransparente
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(raw_rgb, 0.6, heat_color, 0.4, 0)

    # 1. Binarización
    if thr_norm is not None:
        # Si nos dan el umbral en [0,1], lo llevamos a [0,255]
        t = int(np.clip(thr_norm, 0, 1) * 255)
        _, mask = cv2.threshold(heat_u8_for_bin, t, 255, cv2.THRESH_BINARY)
    else:
        # Heurística: percentil 98 dentro de la zona válida
        if np.any(heat_u8_for_bin > 0):
            t = int(np.percentile(heat_u8_for_bin[heat_u8_for_bin > 0], 98))
        else:
            t = 255
        _, mask = cv2.threshold(heat_u8_for_bin, t, 255, cv2.THRESH_BINARY)

    # Limpieza morfológica (ruido/pequeñas discontinuidades)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    # CÁLCULO DE ÁREA REAL (Basado en píxeles)
    # Esto asegura que si hay anomalía detectada, el área no sea 0 aunque no haya
    # contornos grandes tras el filtrado de `AREA_MIN`.
    defect_area_total_px = float(np.count_nonzero(mask))

    # Contornos → polígonos simplificados + dibujo en overlay
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[List[List[int]]] = []
    areas_px: List[float] = []
    defect_area_max_px = 0.0

    for c in cnts:
        area = float(cv2.contourArea(c))
        defect_area_max_px = max(defect_area_max_px, area)

        # Filtramos visualmente para no saturar la UI con ruido, pero el área total ya fue contada arriba
        if area < area_min:
            continue

        approx = cv2.approxPolyDP(c, epsilon=2.0, closed=True)
        polys.append(approx.squeeze(1).tolist())
        areas_px.append(area)
        cv2.polylines(overlay, [approx], True, (0, 255, 0), 2)

    # (Opcional) Dibuja borde de la ROI en amarillo para referencia visual
    if roi_mask is not None:
        border = cv2.Canny(roi_mask, 0, 1)
        overlay[border > 0] = (0, 255, 255)

    # Guardar archivos en disco
    ov_path = os.path.join(overlays_dir, f"{base_name}_overlay.png")
    heat_path = os.path.join(overlays_dir, f"{base_name}_heat.png")
    mask_path = os.path.join(overlays_dir, f"{base_name}_mask.png")
    cv2.imwrite(ov_path, overlay)
    cv2.imwrite(heat_path, heat_color)
    cv2.imwrite(mask_path, mask)

    # Construir URLs públicas
    ov_url = f"/static/{OVERLAYS_SUBDIR}/{os.path.basename(ov_path)}"
    heat_url = f"/static/{OVERLAYS_SUBDIR}/{os.path.basename(heat_path)}"
    mask_url = f"/static/{OVERLAYS_SUBDIR}/{os.path.basename(mask_path)}"

    return (
        ov_path, heat_path, mask_path,
        polys, areas_px,
        ov_url, heat_url, mask_url,
        defect_area_total_px, defect_area_max_px,
        mask
    )


# =======================
# Métricas extra: IoU y logging
# =======================


def compute_iou_masks(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calcula IoU (Intersection over Union) entre dos máscaras binarias.

    Args:
        pred_mask: máscara predicha (uint8, 0/255).
        gt_mask  : máscara ground truth (uint8, 0/255).

    Si las formas difieren, la máscara GT se reescala a la forma de la predicha.

    Returns:
        float en [0,1]. Si la unión es 0, devuelve 0.0.
    """
    # Ajustar tamaño si es necesario
    if pred_mask.shape != gt_mask.shape:
        gt_mask = cv2.resize(
            gt_mask,
            (pred_mask.shape[1], pred_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    # Binarizar por seguridad
    _, pred_bin = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)
    _, gt_bin = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)

    pred_bool = pred_bin > 0
    gt_bool = gt_bin > 0

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    if union == 0:
        return 0.0
    return float(intersection / union)


def log_prediction(source, filename, score, threshold, is_anomaly, area_tot, area_max, iou):
    """
    Registra cada predicción en un CSV.

    - `source` permite distinguir si viene de /predict o de /predict_batch.
    - `iou` puede ser None; en ese caso se deja en blanco.
    """
    header = ["timestamp", "source", "filename", "score", "threshold",
              "is_anomaly", "area_total", "area_max", "iou"]
    write_header = not os.path.exists(PREDICTION_LOG_PATH)
    row = [
        datetime.now().isoformat(), source, filename, score, threshold,
        int(is_anomaly), area_tot, area_max, "" if iou is None else iou
    ]
    with open(PREDICTION_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


# =======================
# Carga de artefactos (startup)
# =======================
BACKBONE = None
HOOK2 = None
HOOK3 = None
KNN = None


def load_knn(artifacts_dir: str, k: int) -> NearestNeighbors:
    """
    Carga el memory bank `memory_bank_core.npz` y ajusta un KNN (promedio de distancias).

    - Espera la clave `bank` en el .npz con shape [N, C].
    - Normaliza L2 fila a fila para estabilidad numérica.
    """
    mb_path = os.path.join(artifacts_dir, "memory_bank_core.npz")
    if not os.path.exists(mb_path):
        raise RuntimeError(f"No existe memory bank: {mb_path}")
    data = np.load(mb_path, allow_pickle=True)
    bank = data["bank"].astype(np.float32)
    bank = l2_normalize_rows(bank)
    # n_jobs=-1 → usa todos los cores disponibles
    knn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1)
    knn.fit(bank)
    return knn


@app.on_event("startup")
def _on_startup():
    """
    Inicializa backbone, hooks, KNN y máscara ROI al levantar el servicio.

    Se imprime la configuración efectiva por consola para depuración.
    """
    global BACKBONE, HOOK2, HOOK3, KNN, ROI_MASK
    BACKBONE, HOOK2, HOOK3 = build_backbone()
    KNN = load_knn(ARTIFACTS_DIR, KNN_K)
    ROI_MASK = build_roi_mask(IMG_SIZE)
    print(f"[startup] Ready. IMG_SIZE={IMG_SIZE} | KNN_K={KNN_K} | THRESHOLD={THRESHOLD}")


@app.get("/health")
def health():
    """Endpoint simple de salud del servicio."""
    return {"status": "ok", "threshold": THRESHOLD}


# =======================
# Endpoints
# =======================
@app.get("/", include_in_schema=False)
def root():
    """
    Sirve un `templates/index.html` básico si existe (frontend).

    Si no existe, devuelve un JSON con un mensaje.
    """
    index_path = os.path.join(BASE_DIR, "templates", "index.html")
    if not os.path.exists(index_path):
        return {"detail": "templates/index.html no encontrado"}
    return FileResponse(index_path)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    thr: Optional[float] = Query(None),
    mode: Optional[str] = Query(None),
    gt_mask: Optional[UploadFile] = File(None),
    source: Optional[str] = Query(None),
):
    """
    Predicción para una sola imagen.

    - `thr`: permite override del threshold vía query param.
    - `mode`: "sensitive" (más sensible) o "strict" (más estricto) ajustan THRESHOLD base.
    - `gt_mask`: imagen opcional de máscara ground truth para calcular IoU real.
    """
    img_bgr = imread_from_upload(file)
    img_gray = bgr_to_gray_256(img_bgr, IMG_SIZE)

    heat, heat_norm, hmin, hmax, score = anomaly_map_and_score(
        img_gray, BACKBONE, HOOK2, HOOK3, KNN, stride=PATCH_STRIDE, roi_mask=ROI_MASK
    )

    # --- Cálculo del threshold efectivo ---
    # Partimos del THRESHOLD global y lo modificamos según `mode` y/o `thr`.
    threshold = THRESHOLD
    if mode == "sensitive":
        threshold *= 0.8
    elif mode == "strict":
        threshold *= 1.2
    if thr is not None:
        threshold = float(thr)

    is_anomaly = bool(score > threshold)

    # Inicializar variables de salida
    overlay_url = None
    polygons = []
    areas_px = []
    defect_area_total_px = 0.0
    defect_area_max_px = 0.0
    pred_mask_array = None  # Para guardar la mascara en memoria

    base_name = os.path.splitext(os.path.basename(file.filename or "upload"))[0].replace(" ", "_")

    if SAVE_VIS:
        # Thr en la escala normalizada [0,1] usando hmin/hmax del caso
        thr_norm = (threshold - hmin) / (hmax - hmin + 1e-8)
        (
            _, _, _, polys, areas_px, ov_url, _, _,
            total_area, max_area, mask_array
        ) = save_visuals_and_polys(
            img_gray, heat_norm, area_min=AREA_MIN, base_name=base_name,
            thr_norm=thr_norm, roi_mask=ROI_MASK
        )
        overlay_url = ov_url
        defect_area_total_px = total_area
        defect_area_max_px = max_area
        pred_mask_array = mask_array

        # Solo enviamos polígonos al frontend si realmente es anomalía
        if is_anomaly:
            polygons = polys

    # --- LÓGICA DE IOU ---
    iou_value: Optional[float] = None
    roi_pixels = get_roi_area_pixels()

    # 1. Si hay GT, calculamos IoU real (predicción vs GT)
    if gt_mask is not None and pred_mask_array is not None:
        gt_bytes = gt_mask.file.read()
        if gt_bytes:
            gt_arr = np.frombuffer(gt_bytes, dtype=np.uint8)
            gt_img = cv2.imdecode(gt_arr, cv2.IMREAD_GRAYSCALE)
            if gt_img is not None:
                iou_value = compute_iou_masks(pred_mask_array, gt_img)

    # 2. Si NO hay GT (o falló carga), calculamos IoU aproximado
    #    IoU ≈ área_defecto_total / área_ROI
    if iou_value is None and roi_pixels > 0:
        if defect_area_total_px > 0:
            iou_value = float(defect_area_total_px) / float(roi_pixels)
            iou_value = min(1.0, iou_value)
        else:
            iou_value = 0.0

    # Log a CSV
    log_prediction(
        source or "single", file.filename or "up",
        score, threshold, is_anomaly,
        defect_area_total_px, defect_area_max_px, iou_value
    )

    return {
        "score": float(score),
        "threshold": float(threshold),
        "is_anomaly": is_anomaly,
        "polygons": polygons,
        "total_defect_area_px": defect_area_total_px,
        "max_defect_area_px": defect_area_max_px,
        "iou": iou_value,
        "overlay_url": overlay_url,
    }


@app.post("/predict_batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    thr: Optional[float] = Form(None),
    mode: Optional[str] = Form(None),
    gt_labels: Optional[List[int]] = Form(
        None,
        description="Etiquetas ground truth (0=normal, 1=defectuosa) en el mismo orden que los archivos."
    ),
):
    """
    Predicción en lote para varias imágenes.

    - Si `gt_labels` es None → solo se devuelven métricas básicas (sin recall).
    - Si `gt_labels` contiene 0/1 → se calcula `recall` a partir de TP y FN:
        recall = TP / (TP + FN)
      donde:
        TP = GT=1 y modelo dice anomalía
        FN = GT=1 y modelo dice normal
    """
    if not files:
        raise HTTPException(status_code=400, detail="Sin archivos")

    # ------------------------------
    # 1) Threshold común a todo el batch
    # ------------------------------
    threshold_base = THRESHOLD
    if mode == "sensitive":
        threshold_base *= 0.8
    elif mode == "strict":
        threshold_base *= 1.2
    if thr is not None:
        threshold_base = float(thr)

    results = []
    roi_pixels = get_roi_area_pixels()

    # Contadores para recall
    tp = 0
    fn = 0

    # Para KPIs de área
    all_areas = []

    for idx, f in enumerate(files):
        # a) Leer imagen y pasar a gris 256x256
        img_bgr = imread_from_upload(f)
        img_gray = bgr_to_gray_256(img_bgr, IMG_SIZE)

        # b) Inferencia PatchCore
        heat, heat_norm, hmin, hmax, score = anomaly_map_and_score(
            img_gray,
            BACKBONE, HOOK2, HOOK3, KNN,
            stride=PATCH_STRIDE,
            roi_mask=ROI_MASK,
        )

        is_anomaly = bool(score > threshold_base)

        base_name = os.path.splitext(f.filename or "upload")[0].replace(" ", "_")

        defect_area_total = 0.0
        defect_area_max = 0.0
        iou_val = 0.0
        ov_url = None
        polys: List[List[List[int]]] = []

        # c) Visuales y áreas
        if SAVE_VIS:
            thr_norm = (threshold_base - hmin) / (hmax - hmin + 1e-8)

            (
                _ov_path,
                _heat_path,
                _mask_path,
                p_list,
                _areas_px,
                url,
                _heat_url,
                _mask_url,
                tot_area,
                mx_area,
                _mask_arr,
            ) = save_visuals_and_polys(
                img_gray,
                heat_norm,
                area_min=AREA_MIN,
                base_name=base_name,
                thr_norm=thr_norm,
                roi_mask=ROI_MASK,
            )

            defect_area_total = float(tot_area)
            defect_area_max = float(mx_area)
            ov_url = url

            if is_anomaly:
                polys = p_list

            all_areas.append(defect_area_total)

            # IoU aproximado: área_defecto_total / área_ROI
            if roi_pixels > 0:
                iou_val = min(1.0, defect_area_total / roi_pixels)

        # d) Cálculo de TP/FN para recall (solo si hay GT)
        if gt_labels is not None and idx < len(gt_labels):
            gt_is_defective = bool(gt_labels[idx])
            if gt_is_defective:
                if is_anomaly:
                    tp += 1
                else:
                    fn += 1

        # nombre "limpio" para el front (sin rutas)
        orig_name = f.filename or "upload"
        safe_name = orig_name.replace("\\", "/").split("/")[-1]

        results.append({
            "idx": idx,
            "filename": safe_name,
            "score": float(score),
            "threshold": float(threshold_base),
            "is_anomaly": is_anomaly,
            "total_defect_area_px": defect_area_total,
            "max_defect_area_px": defect_area_max,
            "iou": iou_val,
            "overlay_url": ov_url,
            "polygons": polys,
        })

    # ------------------------------
    # 2) KPIs globales
    # ------------------------------
    total_images = len(results)
    anomalies = sum(1 for r in results if r["is_anomaly"])
    normals = total_images - anomalies
    defect_rate = float(anomalies) / float(total_images) if total_images > 0 else 0.0

    avg_defect_area_px = float(sum(all_areas) / len(all_areas)) if all_areas else 0.0

    # ---- RECALL ----
    recall: Optional[float] = None
    if gt_labels is not None:
        positives = tp + fn
        print("[DEBUG] gt_labels:", gt_labels, "tp:", tp, "fn:", fn, "positives:", positives)
        if positives > 0:
            recall = tp / positives


    summary = {
        "total_images": total_images,
        "anomalies": anomalies,
        "normals": normals,
        "defect_rate": defect_rate,
        "avg_defect_area_px": avg_defect_area_px,
        "recall": recall,
    }

    return {
        "threshold": float(threshold_base),
        "results": results,
        "summary": summary,
    }
