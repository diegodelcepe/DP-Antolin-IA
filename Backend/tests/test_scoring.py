import numpy as np
import torch
import main as appmod

def test_anomaly_map_and_score_uses_roi(monkeypatch):
    # Hooks y backbone ya mockeados en conftest
    # ROI que deja solo un cuadrado central
    size = appmod.IMG_SIZE
    roi = np.zeros((size,size), np.uint8)
    s = size//4
    roi[s:-s, s:-s] = 255

    # KNN con distancias que generan valores > fuera de ROI; score debe salir del interior
    class KNNConst:
        def kneighbors(self, X, return_distance=True):
            # Distancias mayores al inicio del vector
            d = np.linspace(0.1, 0.9, X.shape[0]*3, dtype=np.float32).reshape(X.shape[0], 3)
            idx = np.zeros_like(d, dtype=int)
            return d, idx

    monkeypatch.setattr(appmod, "KNN", KNNConst(), raising=False)

    # Gray dummy
    g = np.full((size,size), 128, np.uint8)
    heat, heat_norm, hmin, hmax, score = appmod.anomaly_map_and_score(
        g, appmod.BACKBONE, appmod.HOOK2, appmod.HOOK3, appmod.KNN,
        stride=appmod.PATCH_STRIDE, roi_mask=roi
    )
    assert heat.shape == (size, size)
    assert 0.0 <= float(heat_norm.min()) <= float(heat_norm.max()) <= 1.0
    # score es un float
    assert isinstance(score, float)
