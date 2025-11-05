import io
import os
import types
import numpy as np
import cv2
import torch
import pytest
from fastapi.testclient import TestClient
import main as appmod

@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    # Aislar artefactos y estáticos
    monkeypatch.setattr(appmod, "STATIC_DIR", str(tmp_path / "static"), raising=False)
    monkeypatch.setattr(appmod, "OVERLAYS_SUBDIR", "overlays", raising=False)
    (tmp_path / "static").mkdir(parents=True, exist_ok=True)

    # Evitar GPU en tests (si existiera)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)
    monkeypatch.setattr(appmod, "DEVICE", "cpu", raising=False)

    # Evitar leer memory bank real en startup
    yield

@pytest.fixture
def client(monkeypatch):
    # Mockear backbone + hooks + KNN para toda la app
    class DummyHook:
        def __init__(self, feat=None):
            self.feat = feat
        def close(self): pass

    class DummyBackbone(torch.nn.Module):
        def forward(self, x):
            # No hace nada; las feats las daremos por hook
            return torch.zeros(1)

    def fake_build_backbone():
        # Genera mapas de caracteristicas sintéticos y coherentes
        # f2: (B, C2, Hf, Wf) ; f3: (B, C3, Hf/2, Wf/2) — luego se interpola
        C2, C3, Hf, Wf = 16, 8, 16, 16
        h2 = DummyHook()
        h3 = DummyHook()
        def set_feats(x):
            h2.feat = torch.randn(1, C2, Hf, Wf)
            h3.feat = torch.randn(1, C3, Hf//2, Wf//2)
            return torch.zeros(1)
        bb = DummyBackbone()
        bb.forward = set_feats
        return bb, h2, h3

    class DummyKNN:
        def __init__(self): pass
        def kneighbors(self, X, return_distance=True):
            # Distancias deterministas en [0,1]
            rng = np.random.default_rng(123)
            d = rng.random((X.shape[0], 3)).astype(np.float32)
            idx = np.zeros_like(d, dtype=int)
            return d, idx

    monkeypatch.setattr(appmod, "build_backbone", fake_build_backbone, raising=True)
    monkeypatch.setattr(appmod, "load_knn", lambda *a, **k: DummyKNN(), raising=True)

    # Forzar reconstrucción de la app: invocar startup manualmente
    appmod.BACKBONE, appmod.HOOK2, appmod.HOOK3 = appmod.build_backbone()
    appmod.KNN = appmod.load_knn("", 3)
    appmod.ROI_MASK = appmod.build_roi_mask(appmod.IMG_SIZE)

    return TestClient(appmod.app)

@pytest.fixture
def sample_bgr():
    # Imagen BGR 64x64
    img = np.zeros((64, 64, 3), np.uint8)
    cv2.circle(img, (32, 32), 12, (0, 0, 255), -1)  # algo no-gris uniforme
    return img

def encode_png(img):
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return io.BytesIO(buf.tobytes())
