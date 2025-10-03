# main.py
# FastAPI para inferencia PatchCore con ROI (ignorar bordes o máscara),
# umbral consistente para decisión y polígonos, y guardado de overlays
# en static/overlays (configurable).

import os, json, math
from typing import Optional, Tuple, List

import numpy as np
import cv2
import torch
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse  # <- para servir templates/index.html
from dotenv import load_dotenv  # <- NUEVO

# =======================
# Paths y helpers
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _abs(path: str) -> str:
    """Vuelve absoluta una ruta (si es relativa, la resuelve desde BASE_DIR)."""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)

# Cargar .env (si existe) ANTES de leer os.getenv(...)
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"), override=False)

# =======================
# Config por entorno
# =======================
ARTIFACTS_DIR   = _abs(os.getenv("ARTIFACTS_DIR", os.path.join("models", "patchcore")))
STATIC_DIR      = _abs(os.getenv("STATIC_DIR", "static"))
OVERLAYS_SUBDIR = os.getenv("OVERLAYS_SUBDIR", "overlays")  # subcarpeta dentro de STATIC_DIR

CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "config.json")
CONFIG_JSON: dict = {}
if os.path.exists(CONFIG_PATH):
    try:
        CONFIG_JSON = json.load(open(CONFIG_PATH, "r", encoding="utf-8"))
    except Exception:
        CONFIG_JSON = {}

THRESHOLD    = float(os.getenv("THRESHOLD", str(CONFIG_JSON.get("threshold", 0.35))))
IMG_SIZE     = int(os.getenv("IMG_SIZE", "256"))      # Debe coincidir con el memory bank
KNN_K        = int(os.getenv("KNN_K", "3"))
PATCH_STRIDE = int(os.getenv("PATCH_STRIDE", "1"))
SAVE_VIS     = os.getenv("SAVE_VIS", "1") == "1"

# Visual / polígonos
AREA_MIN = int(os.getenv("AREA_MIN", "200"))          # área mínima para contornos

# ROI
IGNORE_BORDER_PCT = float(os.getenv("IGNORE_BORDER_PCT", "0"))  # ej. 8 = recorta 8% cada lado
ROI_PATH_ENV      = os.getenv("ROI_PATH", "")                   # PNG binaria (blanco=ROI)
ROI_PATH          = _abs(ROI_PATH_ENV) if ROI_PATH_ENV else ""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =======================
# FastAPI
# =======================
app = FastAPI(title="PatchCore Anomaly API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# =======================
# Utilidades
# =======================
def imread_from_upload(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Archivo vacío.")
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen.")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def bgr_to_gray_256(img_bgr: np.ndarray, size: int) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return g

def to_tensor_3ch(gray: np.ndarray) -> torch.Tensor:
    x = (gray.astype(np.float32) / 255.0)
    x = np.stack([x, x, x], axis=0)  # 3 canales
    return torch.from_numpy(x).unsqueeze(0).to(DEVICE)

# =======================
# Backbone + Hooks
# =======================
class FeatHook:
    def __init__(self, m):
        self.h = m.register_forward_hook(self._hook)
        self.feat = None
    def _hook(self, m, inp, out):
        self.feat = out.detach()
    def close(self):
        self.h.remove()

def build_backbone() -> Tuple[torch.nn.Module, FeatHook, FeatHook]:
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(DEVICE)
    backbone.eval()
    layers = dict(backbone.named_modules())
    h2 = FeatHook(layers["layer2"])
    h3 = FeatHook(layers["layer3"])
    return backbone, h2, h3

def extract_concat_features(x: torch.Tensor,
                            backbone: torch.nn.Module,
                            h2: FeatHook, h3: FeatHook) -> torch.Tensor:
    with torch.no_grad():
        _ = backbone(x)
    f2 = h2.feat
    f3 = h3.feat
    f3u = torch.nn.functional.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
    fcat = torch.cat([f2, f3u], dim=1).squeeze(0)  # (C, Hf, Wf)
    return fcat

def patchify_feature_map(fmap: torch.Tensor, stride: int = 1) -> torch.Tensor:
    C, H, W = fmap.shape
    if stride <= 1:
        return fmap.permute(1, 2, 0).reshape(H * W, C).contiguous()
    f = fmap[:, ::stride, ::stride]
    h, w = f.shape[-2:]
    return f.permute(1, 2, 0).reshape(h * w, C).contiguous()

def l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / n

# =======================
# ROI helpers
# =======================
ROI_MASK: Optional[np.ndarray] = None  # uint8 (IMG_SIZE,IMG_SIZE) 255=ROI, 0=ignore

def build_roi_mask(img_size: int) -> Optional[np.ndarray]:
    mask = np.ones((img_size, img_size), np.uint8) * 255
    # 1) Ignorar bordes por porcentaje
    if IGNORE_BORDER_PCT > 0:
        m = int(round(img_size * IGNORE_BORDER_PCT / 100.0))
        if m > 0:
            mask[:m, :] = 0
            mask[-m:, :] = 0
            mask[:, :m] = 0
            mask[:, -m:] = 0
    # 2) Máscara desde archivo
    if ROI_PATH and os.path.exists(ROI_PATH):
        m2 = cv2.imread(ROI_PATH, cv2.IMREAD_GRAYSCALE)
        if m2 is not None:
            m2 = cv2.resize(m2, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
            _, m2b = cv2.threshold(m2, 127, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_and(mask, m2b)
    if mask.max() == 0:
        return None
    return mask

# =======================
# Scoring + Visuales
# =======================
def anomaly_map_and_score(gray_img: np.ndarray,
                          backbone: torch.nn.Module,
                          h2: FeatHook, h3: FeatHook,
                          knn: NearestNeighbors,
                          stride: int = PATCH_STRIDE,
                          roi_mask: Optional[np.ndarray] = None
                          ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """
    Retorna:
      heat (crudo), heat_norm [0..1], hmin, hmax, score (máximo dentro de ROI si existe)
    """
    x = to_tensor_3ch(gray_img)
    fcat = extract_concat_features(x, backbone, h2, h3)
    Hf, Wf = fcat.shape[-2:]
    patches = patchify_feature_map(fcat, stride=stride)
    patches = torch.nn.functional.normalize(patches, p=2, dim=1).cpu().numpy()
    dists, _ = knn.kneighbors(patches, return_distance=True)
    ph = dists.mean(axis=1).reshape(
        Hf if stride <= 1 else math.ceil(Hf / stride),
        Wf if stride <= 1 else math.ceil(Wf / stride)
    ).astype(np.float32)

    heat = cv2.resize(ph, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    hmin, hmax = float(heat.min()), float(heat.max())
    heat_norm = (heat - hmin) / (hmax - hmin + 1e-8)

    # Score solo dentro de la ROI (si existe)
    if roi_mask is not None:
        heat_for_score = heat.copy()
        heat_for_score[roi_mask == 0] = hmin - 1.0
        score = float(heat_for_score.max())
    else:
        score = float(heat.max())

    return heat, heat_norm, hmin, hmax, score

def save_visuals_and_polys(img_gray: np.ndarray, heat_norm: np.ndarray,
                           area_min: int,
                           base_name: str,
                           thr_norm: Optional[float] = None,
                           roi_mask: Optional[np.ndarray] = None
                           ) -> Tuple[str, str, str, List[List[List[int]]], str, str, str]:
    """
    Genera overlay/heat/mask y polígonos; si thr_norm está presente, usa ese umbral (0..1).
    Aplica ROI para máscara/polígonos. Guarda en static/OVERLAYS_SUBDIR y devuelve URLs públicas.
    """
    overlays_dir = os.path.join(STATIC_DIR, OVERLAYS_SUBDIR)
    os.makedirs(overlays_dir, exist_ok=True)

    raw_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    heat_u8 = (heat_norm * 255).astype(np.uint8)

    # ROI para binarización (no para colormap)
    heat_u8_for_bin = cv2.bitwise_and(heat_u8, heat_u8, mask=roi_mask) if roi_mask is not None else heat_u8

    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(raw_rgb, 0.6, heat_color, 0.4, 0)

    if thr_norm is not None:
        t = int(np.clip(thr_norm, 0, 1) * 255)
        _, mask = cv2.threshold(heat_u8_for_bin, t, 255, cv2.THRESH_BINARY)
    else:
        t = int(np.percentile(heat_u8_for_bin[heat_u8_for_bin > 0], 98)) if np.any(heat_u8_for_bin > 0) else 255
        _, mask = cv2.threshold(heat_u8_for_bin, t, 255, cv2.THRESH_BINARY)

    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[List[List[int]]] = []
    for c in cnts:
        if cv2.contourArea(c) < area_min:
            continue
        approx = cv2.approxPolyDP(c, epsilon=2.0, closed=True)
        polys.append(approx.squeeze(1).tolist())
        cv2.polylines(overlay, [approx], True, (0, 255, 0), 2)

    # (opcional) borde de la ROI
    if roi_mask is not None:
        border = cv2.Canny(roi_mask, 0, 1)
        overlay[border > 0] = (0, 255, 255)

    ov_path   = os.path.join(overlays_dir, f"{base_name}_overlay.png")
    heat_path = os.path.join(overlays_dir, f"{base_name}_heat.png")
    mask_path = os.path.join(overlays_dir, f"{base_name}_mask.png")
    cv2.imwrite(ov_path, overlay)
    cv2.imwrite(heat_path, heat_color)
    cv2.imwrite(mask_path, mask)

    # URLs públicas
    ov_url   = f"/static/{OVERLAYS_SUBDIR}/{os.path.basename(ov_path)}"
    heat_url = f"/static/{OVERLAYS_SUBDIR}/{os.path.basename(heat_path)}"
    mask_url = f"/static/{OVERLAYS_SUBDIR}/{os.path.basename(mask_path)}"

    return ov_path, heat_path, mask_path, polys, ov_url, heat_url, mask_url

# =======================
# Carga de artefactos
# =======================
BACKBONE: Optional[torch.nn.Module] = None
HOOK2: Optional[FeatHook] = None
HOOK3: Optional[FeatHook] = None
KNN: Optional[NearestNeighbors] = None
ROI_MASK: Optional[np.ndarray] = None

def load_knn(artifacts_dir: str, k: int) -> NearestNeighbors:
    mb_path = os.path.join(artifacts_dir, "memory_bank_core.npz")
    if not os.path.exists(mb_path):
        raise RuntimeError(f"No existe memory bank: {mb_path}")
    data = np.load(mb_path, allow_pickle=True)
    bank = data["bank"].astype(np.float32)
    bank = l2_normalize_rows(bank)
    knn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1)
    knn.fit(bank)
    return knn

@app.on_event("startup")
def _on_startup():
    global BACKBONE, HOOK2, HOOK3, KNN, ROI_MASK
    BACKBONE, HOOK2, HOOK3 = build_backbone()
    KNN = load_knn(ARTIFACTS_DIR, KNN_K)
    ROI_MASK = build_roi_mask(IMG_SIZE)
    print(
        f"[startup] Device={DEVICE} | IMG_SIZE={IMG_SIZE} | KNN_K={KNN_K} | THRESHOLD={THRESHOLD} | "
        f"IGNORE_BORDER_PCT={IGNORE_BORDER_PCT} | ROI_PATH={'set' if ROI_PATH else 'none'} | "
        f"STATIC_DIR={STATIC_DIR} | OVERLAYS_SUBDIR={OVERLAYS_SUBDIR}"
    )

# =======================
# Endpoints
# =======================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "img_size": IMG_SIZE,
        "knn_k": KNN_K,
        "threshold": THRESHOLD,
        "ignore_border_pct": IGNORE_BORDER_PCT,
        "roi_path": ROI_PATH if ROI_PATH else None
    }

@app.get("/", include_in_schema=False)
def root():
    """Sirve el frontend estático."""
    index_path = os.path.join(BASE_DIR, "templates", "index.html")
    if not os.path.exists(index_path):
        return {"detail": "templates/index.html no encontrado"}
    return FileResponse(index_path)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    thr: Optional[float] = Query(None, description="Umbral manual (sobrescribe config/env)"),
    mode: Optional[str] = Query(None, description="sensitive (umbral*0.8) | strict (umbral*1.2)")
):
    img_bgr = imread_from_upload(file)
    img_gray = bgr_to_gray_256(img_bgr, IMG_SIZE)

    heat, heat_norm, hmin, hmax, score = anomaly_map_and_score(
        img_gray, BACKBONE, HOOK2, HOOK3, KNN, stride=PATCH_STRIDE, roi_mask=ROI_MASK
    )

    # Umbral efectivo
    threshold = THRESHOLD
    if mode == "sensitive":
        threshold *= 0.8
    elif mode == "strict":
        threshold *= 1.2
    if thr is not None:
        threshold = float(thr)

    is_anomaly = bool(score > threshold)

    overlay_url = None
    polygons: List[List[List[int]]] = []

    base_name = os.path.splitext(os.path.basename(file.filename or "upload"))[0]
    base_name = base_name.replace(" ", "_")

    if SAVE_VIS:
        thr_norm = (threshold - hmin) / (hmax - hmin + 1e-8)
        _, _, _, polys, ov_url, _, _ = save_visuals_and_polys(
            img_gray, heat_norm,
            area_min=AREA_MIN,
            base_name=base_name,
            thr_norm=thr_norm,
            roi_mask=ROI_MASK
        )
        overlay_url = ov_url
        if is_anomaly:
            polygons = polys

    return {
        "score": float(score),
        "threshold": float(threshold),
        "is_anomaly": is_anomaly,
        "polygons": polygons,
        "overlay_url": overlay_url
    }
