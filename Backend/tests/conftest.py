# tests/conftest.py
import io
import numpy as np
import cv2
import torch
import pytest
from fastapi.testclient import TestClient

import main as appmod  # importa tu app desde Backend/main.py

@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    # Aislar directorio estático para no ensuciar el repo
    static_dir = tmp_path / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(appmod, "STATIC_DIR", str(static_dir), raising=False)
    monkeypatch.setattr(appmod, "OVERLAYS_SUBDIR", "overlays", raising=False)

    # Forzar CPU en tests
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)
    monkeypatch.setattr(appmod, "DEVICE", "cpu", raising=False)
    yield

@pytest.fixture
def client(monkeypatch):
    # Backbone + hooks falsos y KNN determinista
    class DummyHook:
        def __init__(self): self.feat = None
        def close(self): pass

    class DummyBackbone(torch.nn.Module):
        def forward(self, x):  # no se usa; lo sobreescribiremos
            return torch.zeros(1)

    def fake_build_backbone():
        C2, C3, Hf, Wf = 16, 8, 16, 16
        h2, h3 = DummyHook(), DummyHook()
        bb = DummyBackbone()
        def set_feats(x):
            # features sintéticas pero coherentes
            h2.feat = torch.randn(1, C2, Hf, Wf)
            h3.feat = torch.randn(1, C3, Hf//2, Wf//2)
            return torch.zeros(1)
        bb.forward = set_feats
        return bb, h2, h3

    class DummyKNN:
        def kneighbors(self, X, return_distance=True):
            rng = np.random.default_rng(123)
            d = rng.random((X.shape[0], 3)).astype(np.float32)
            idx = np.zeros_like(d, dtype=int)
            return d, idx

    # Parchea constructores si existen; si no, no falla
    monkeypatch.setattr(appmod, "build_backbone", fake_build_backbone, raising=False)
    monkeypatch.setattr(appmod, "load_knn", lambda *a, **k: DummyKNN(), raising=False)

    # (Re)inicializa globals usados por la app
    appmod.BACKBONE, appmod.HOOK2, appmod.HOOK3 = fake_build_backbone()
    appmod.KNN = DummyKNN()
    appmod.ROI_MASK = appmod.build_roi_mask(appmod.IMG_SIZE)

    return TestClient(appmod.app)

@pytest.fixture
def sample_bgr():
    img = np.zeros((64, 64, 3), np.uint8)
    cv2.circle(img, (32, 32), 12, (0, 0, 255), -1)
    return img

@pytest.fixture
def png_bytes(sample_bgr):
    ok, buf = cv2.imencode(".png", sample_bgr)
    assert ok
    return io.BytesIO(buf.tobytes())
