"""
PatchCore Anomaly API (FastAPI)
--------------------------------
Versión comentada y explicada del servicio de inferencia para inspección de piezas
con el método tipo PatchCore + KNN. Incluye:

- Carga de configuración desde variables de entorno y/o `models/patchcore/config.json`.
- Preprocesamiento a escala fija (IMG_SIZE) y conversión a tensor 3 canales (gris replicado).
- Extracción de características desde ResNet-18 (layers 2 y 3), con *hooks*.
- Normalización L2 por parches y búsqueda de vecinos más cercanos (KNN) en un *memory bank*.
- Mapa de calor de anormalidad (distancia promedio a K vecinos) y puntuación global.
- Soporte a ROI: ignorar bordes por porcentaje y/o usar una máscara binaria externa.
- Generación de *overlays* (mapa de calor + contornos por umbral) y guardado en `static/overlays`.
- Endpoints `/predict` (1 imagen) y `/predict_batch` (varias imágenes) con opciones de umbral.

⚠️ NOTA: Esta versión sólo añade comentarios y docstrings. La lógica y el comportamiento son los mismos.
"""

# =======================
# Imports
# =======================
import os, json, math
from typing import Optional, Tuple, List

import numpy as np
import cv2
import torch
import torchvision.models as models
from sklearn.neighbors import NearestNeighbors
from statistics import mean

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from dotenv import load_dotenv  # Carga variables desde .env si existe

# =======================
# Paths y helpers
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _abs(path: str) -> str:
    """Devuelve una ruta absoluta. Si `path` es relativa, la resuelve desde `BASE_DIR`."""
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)

# Cargar .env (si existe) ANTES de leer os.getenv(...)
# `override=False` evita sobreescribir variables ya presentes en el entorno del proceso
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"), override=False)

# =======================
# Config por entorno
# =======================
# Ruta base de artefactos del modelo (memory bank, config.json, etc.)
ARTIFACTS_DIR   = _abs(os.getenv("ARTIFACTS_DIR", os.path.join("models", "patchcore")))
# Directorio para servir archivos estáticos (overlays, etc.)
STATIC_DIR      = _abs(os.getenv("STATIC_DIR", "static"))
# Subcarpeta donde se guardarán las imágenes generadas (overlays, heatmaps, masks)
OVERLAYS_SUBDIR = os.getenv("OVERLAYS_SUBDIR", "overlays")

# Cargar JSON de configuración si existe (permite fijar threshold, etc.)
CONFIG_PATH = os.path.join(ARTIFACTS_DIR, "config.json")
CONFIG_JSON: dict = {}
if os.path.exists(CONFIG_PATH):
    try:
        CONFIG_JSON = json.load(open(CONFIG_PATH, "r", encoding="utf-8"))
    except Exception:
        # Silencioso a propósito: si falla la lectura, continuamos con valores por defecto
        CONFIG_JSON = {}

# Hiperparámetros / ajustes
THRESHOLD    = float(os.getenv("THRESHOLD", str(CONFIG_JSON.get("threshold", 0.35))))
IMG_SIZE     = int(os.getenv("IMG_SIZE", "256"))      # Debe coincidir con el memory bank
KNN_K        = int(os.getenv("KNN_K", "3"))
PATCH_STRIDE = int(os.getenv("PATCH_STRIDE", "1"))     # Submuestreo del mapa de features para KNN
SAVE_VIS     = os.getenv("SAVE_VIS", "1") == "1"        # Guardar visualizaciones (overlay/heat/mask)

# Visual / polígonos
AREA_MIN = int(os.getenv("AREA_MIN", "200"))           # Área mínima (px) para considerar un contorno

# ROI: dos mecanismos (acumulables)
# 1) IGNORE_BORDER_PCT: ignora un porcentaje desde cada borde (en la imagen final IMG_SIZE×IMG_SIZE)
# 2) ROI_PATH: ruta a una máscara binaria (PNG) donde blanco=ROI válido y negro=ignorar
IGNORE_BORDER_PCT = float(os.getenv("IGNORE_BORDER_PCT", "0"))
ROI_PATH_ENV      = os.getenv("ROI_PATH", "")
ROI_PATH          = _abs(ROI_PATH_ENV) if ROI_PATH_ENV else ""

# Selección de dispositivo para PyTorch
device_available = torch.cuda.is_available()
DEVICE = "cuda" if device_available else "cpu"

# =======================
# FastAPI (app + CORS + estáticos)
# =======================
app = FastAPI(title="PatchCore Anomaly API", version="1.2.0")

# CORS abierto (útil para prototipos / pruebas). En producción, restringir dominios.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Asegurar que el directorio de estáticos exista y montarlo en /static
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# =======================
# Utilidades de imagen y tensores
# =======================

def imread_from_upload(file: UploadFile) -> np.ndarray:
    """Lee bytes de un UploadFile de FastAPI y devuelve BGR uint8.
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
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:  # BGRA → BGR (descarta canal alpha)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def bgr_to_gray_256(img_bgr: np.ndarray, size: int) -> np.ndarray:
    """Convierte BGR→GRIS y redimensiona a `size×size` con INTER_AREA.
    Devuelve `np.uint8` [0..255].
    """
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    return g


def to_tensor_3ch(gray: np.ndarray) -> torch.Tensor:
    """Normaliza a [0,1], replica el canal gris a 3 canales y devuelve tensor [1,3,H,W] en `DEVICE`."""
    x = (gray.astype(np.float32) / 255.0)
    x = np.stack([x, x, x], axis=0)  # 3 canales idénticos (ResNet espera 3 canales)
    return torch.from_numpy(x).unsqueeze(0).to(DEVICE)

# =======================
# Backbone + Hooks (extracción de features)
# =======================
class FeatHook:
    """Pequeño contenedor para registrar un *forward hook* y almacenar la salida intermedia."""
    def __init__(self, m):
        self.h = m.register_forward_hook(self._hook)
        self.feat = None
    def _hook(self, m, inp, out):
        # Guardar la activación (sin gradiente)
        self.feat = out.detach()
    def close(self):
        # Liberar el hook cuando no se necesite
        self.h.remove()


def build_backbone() -> Tuple[torch.nn.Module, FeatHook, FeatHook]:
    """Construye ResNet-18 pre-entrenada y registra hooks en `layer2` y `layer3`.
    Devuelve: (backbone, hook_layer2, hook_layer3).
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
    h3: FeatHook
) -> torch.Tensor:
    """Realiza forward para poblar hooks y concatena features (layer2 || upsample(layer3)).
    Salida: tensor [C_concat, Hf, Wf].
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
    """Convierte un mapa de features [C,H,W] en una matriz de parches [N_patches, C].
    - Si `stride>1`, submuestrea espacialmente H y W para acelerar KNN a costa de resolución.
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
ROI_MASK: Optional[np.ndarray] = None  # Máscara binaria en escala IMG_SIZE×IMG_SIZE (255=ROI, 0=ignorar)

def build_roi_mask(img_size: int) -> Optional[np.ndarray]:
    """Crea una máscara ROI combinando dos fuentes:
    1) Borde ignorado por porcentaje (cuadro más pequeño válido).
    2) Máscara binaria externa opcional (PNG), reescalada y binarizada.
    Devuelve `None` si la máscara final es toda cero (i.e., no hay píxeles válidos).
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
        return None
    return mask

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
    roi_mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Calcula mapa de anormalidad y score global.

    Pasos:
    - Convierte la imagen gris a tensor 3ch y extrae features concatenadas (layer2+layer3↑).
    - *Patchify* y normaliza L2 cada vector de parche.
    - KNN: para cada parche, distancia promedio a sus K vecinos más cercanos del memory bank.
    - Reescala a IMG_SIZE y normaliza [0..1] (guardando hmin/hmax para revertir umbrales a espacio original).
    - `score` = máx(heat) *dentro de la ROI* si existe ROI, si no, máx global de heat.

    Retorna:
      heat (float32 crudo), heat_norm [0..1], hmin, hmax, score (float en escala `heat`).
    """
    x = to_tensor_3ch(gray_img)
    fcat = extract_concat_features(x, backbone, h2, h3)
    Hf, Wf = fcat.shape[-2:]

    # Extraer parches y normalizar L2 (PyTorch→NumPy)
    patches = patchify_feature_map(fcat, stride=stride)
    patches = torch.nn.functional.normalize(patches, p=2, dim=1).cpu().numpy()

    # Distancias KNN (promedio sobre K) → mapa [Hf',Wf']
    dists, _ = knn.kneighbors(patches, return_distance=True)
    ph = dists.mean(axis=1).reshape(
        Hf if stride <= 1 else math.ceil(Hf / stride),
        Wf if stride <= 1 else math.ceil(Wf / stride)
    ).astype(np.float32)

    # Reescalar al tamaño de la imagen final (IMG_SIZE×IMG_SIZE)
    heat = cv2.resize(ph, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    hmin, hmax = float(heat.min()), float(heat.max())
    heat_norm = (heat - hmin) / (hmax - hmin + 1e-8)

    # Score (máximo) restringido a ROI si existe
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
    roi_mask: Optional[np.ndarray] = None
) -> Tuple[str, str, str, List[List[List[int]]], List[float], str, str, str]:
    """Genera y guarda visualizaciones y polígonos de defectos.

    Entradas:
      - `heat_norm`: mapa [0..1] (normalizado con hmin/hmax del caso).
      - `thr_norm`: umbral en [0..1] para binarizar (si None, usa percentil 98 de ROI>0).
      - `roi_mask`: si se provee, sólo se binariza dentro de ROI (para evitar falsos contornos fuera).

    Salidas:
      - Rutas absolutas a overlay, heat y mask.
      - Lista de polígonos (cada uno es lista de [x,y]).
      - Lista de áreas en píxeles por polígono.
      - URLs públicas `/static/...` equivalentes a los archivos.
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

    # Binarización según umbral normalizado
    if thr_norm is not None:
        t = int(np.clip(thr_norm, 0, 1) * 255)
        _, mask = cv2.threshold(heat_u8_for_bin, t, 255, cv2.THRESH_BINARY)
    else:
        # Heurística: percentil 98 dentro de la zona válida
        t = int(np.percentile(heat_u8_for_bin[heat_u8_for_bin > 0], 98)) if np.any(heat_u8_for_bin > 0) else 255
        _, mask = cv2.threshold(heat_u8_for_bin, t, 255, cv2.THRESH_BINARY)

    # Limpieza morfológica (ruido/pequeñas discontinuidades)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    # Contornos → polígonos simplificados + dibujo en overlay
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[List[List[int]]] = []
    areas_px: List[float] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < area_min:
            continue  # descarta manchas pequeñas/ruido
        approx = cv2.approxPolyDP(c, epsilon=2.0, closed=True)
        polys.append(approx.squeeze(1).tolist())
        areas_px.append(float(area))
        cv2.polylines(overlay, [approx], True, (0, 255, 0), 2)

    # (Opcional) Dibuja borde de la ROI en amarillo para referencia visual
    if roi_mask is not None:
        border = cv2.Canny(roi_mask, 0, 1)
        overlay[border > 0] = (0, 255, 255)

    # Guardar archivos en disco
    ov_path   = os.path.join(overlays_dir, f"{base_name}_overlay.png")
    heat_path = os.path.join(overlays_dir, f"{base_name}_heat.png")
    mask_path = os.path.join(overlays_dir, f"{base_name}_mask.png")
    cv2.imwrite(ov_path, overlay)
    cv2.imwrite(heat_path, heat_color)
    cv2.imwrite(mask_path, mask)

    # Construir URLs públicas
    ov_url   = f"/static/{OVERLAYS_SUBDIR}/{os.path.basename(ov_path)}"
    heat_url = f"/static/{OVERLAYS_SUBDIR}/{os.path.basename(heat_path)}"
    mask_url = f"/static/{OVERLAYS_SUBDIR}/{os.path.basename(mask_path)}"

    return ov_path, heat_path, mask_path, polys, areas_px, ov_url, heat_url, mask_url

# =======================
# Carga de artefactos (startup)
# =======================
BACKBONE: Optional[torch.nn.Module] = None
HOOK2: Optional[FeatHook] = None
HOOK3: Optional[FeatHook] = None
KNN: Optional[NearestNeighbors] = None
ROI_MASK: Optional[np.ndarray] = None


def load_knn(artifacts_dir: str, k: int) -> NearestNeighbors:
    """Carga el memory bank `memory_bank_core.npz` y ajusta un KNN (promedio de distancias).
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
    """Inicializa backbone, hooks, KNN y máscara ROI al levantar el servicio."""
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
    """Endpoint de salud: devuelve configuración efectiva del servicio."""
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
    """Sirve un `templates/index.html` básico si existe (frontend muy simple)."""
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
    """Predicción para una sola imagen.
    - `thr`: umbral absoluto en la escala del mapa `heat` (antes de normalizar).
    - `mode`: atajo para ajustar el threshold base (0.8× o 1.2×) si no se pasa `thr`.
    - Devuelve score, umbral usado, flag binario, polígonos (si se guardan vis) y URL del overlay.
    """
    # 1) Leer imagen y preparar gris 256×256
    img_bgr = imread_from_upload(file)
    img_gray = bgr_to_gray_256(img_bgr, IMG_SIZE)

    # 2) Calcular heatmap y score global
    heat, heat_norm, hmin, hmax, score = anomaly_map_and_score(
        img_gray, BACKBONE, HOOK2, HOOK3, KNN, stride=PATCH_STRIDE, roi_mask=ROI_MASK
    )

    # 3) Determinar umbral efectivo (prioridad: mode → thr explícito)
    threshold = THRESHOLD
    if mode == "sensitive":
        threshold *= 0.8
    elif mode == "strict":
        threshold *= 1.2
    if thr is not None:
        threshold = float(thr)

    # 4) Clasificación binaria
    is_anomaly = bool(score > threshold)

    overlay_url = None
    polygons: List[List[List[int]]] = []
    areas_px: List[float] = []

    # Nombre base para archivos (limpia espacios)
    base_name = os.path.splitext(os.path.basename(file.filename or "upload"))[0]
    base_name = base_name.replace(" ", "_")

    # 5) Visualizaciones opcionales (y polígonos sólo si es anómalo)
    if SAVE_VIS:
        # Convertimos el threshold a la escala normalizada [0..1]
        thr_norm = (threshold - hmin) / (hmax - hmin + 1e-8)
        _, _, _, polys, areas_px, ov_url, _, _ = save_visuals_and_polys(
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
        "polygon_areas_px": areas_px,   # devuelto si SAVE_VIS
        "overlay_url": overlay_url
    }


@app.post("/predict_batch")
async def predict_batch(
    files: List[UploadFile] = File(..., description="Varias imágenes"),
    thr: Optional[float] = Query(None, description="Umbral manual (sobrescribe config/env)"),
    mode: Optional[str] = Query(None, description="sensitive (0.8×) | strict (1.2×)")
):
    """Predicción por lotes.
    - Aplica el *mismo* umbral a todas las imágenes del batch (calculado desde `THRESHOLD`, `mode` y/o `thr`).
    - Devuelve resumen agregado (tasas y área promedio) y la lista de resultados individuales.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No se enviaron archivos.")

    # 1) Umbral base común a todo el batch
    threshold_base = THRESHOLD
    if mode == "sensitive":
        threshold_base *= 0.8
    elif mode == "strict":
        threshold_base *= 1.2
    if thr is not None:
        threshold_base = float(thr)

    results = []
    n_anom = 0
    n_norm = 0
    all_defect_areas = []  # Áreas de todos los polígonos de todas las imágenes

    for f in files:
        # a) Leer/convertir
        img_bgr = imread_from_upload(f)
        img_gray = bgr_to_gray_256(img_bgr, IMG_SIZE)

        # b) Mapa y score
        heat, heat_norm, hmin, hmax, score = anomaly_map_and_score(
            img_gray, BACKBONE, HOOK2, HOOK3, KNN, stride=PATCH_STRIDE, roi_mask=ROI_MASK
        )

        # c) Decisión binaria con el mismo umbral para todo el lote
        threshold = threshold_base
        is_anomaly = bool(score > threshold)

        # d) Visual + polígonos (sólo se informan si anómalo)
        overlay_url = None
        polygons: List[List[List[int]]] = []
        poly_areas: List[float] = []

        base_name = os.path.splitext(os.path.basename(f.filename or "upload"))[0]
        base_name = base_name.replace(" ", "_")

        if SAVE_VIS:
            thr_norm = (threshold - hmin) / (hmax - hmin + 1e-8)
            _, _, _, polys, areas_px, ov_url, _, _ = save_visuals_and_polys(
                img_gray, heat_norm,
                area_min=AREA_MIN,
                base_name=base_name,
                thr_norm=thr_norm,
                roi_mask=ROI_MASK
            )
            overlay_url = ov_url
            if is_anomaly:
                polygons = polys
                poly_areas = areas_px
                all_defect_areas.extend(areas_px)

        # e) Contadores agregados
        if is_anomaly:
            n_anom += 1
        else:
            n_norm += 1

        results.append({
            "filename": f.filename,
            "score": float(score),
            "threshold": float(threshold),
            "is_anomaly": is_anomaly,
            "polygons": polygons,
            "polygon_areas_px": poly_areas,
            "overlay_url": overlay_url
        })

    # 2) Resumen del lote
    n_total = len(results)
    defect_rate = (n_anom / n_total) if n_total else 0.0
    avg_defect_area = mean(all_defect_areas) if all_defect_areas else 0.0

    summary = {
        "total_images": n_total,
        "anomalies": n_anom,
        "normals": n_norm,
        "defect_rate": defect_rate,             # proporción 0..1
        "avg_defect_area_px": avg_defect_area   # píxeles en imagen IMG_SIZE×IMG_SIZE
    }
    return {"summary": summary, "results": results}

