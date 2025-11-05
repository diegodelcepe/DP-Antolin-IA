import io
import numpy as np
import cv2
import main as appmod

def _make_image_bytes():
    img = np.zeros((128,128,3), np.uint8)
    cv2.circle(img, (64,64), 20, (0,0,255), -1)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "device" in data and "img_size" in data

def test_predict_ok(client, monkeypatch):
    # Forzar SAVE_VIS
    monkeypatch.setattr(appmod, "SAVE_VIS", True, raising=False)

    files = {"file": ("f.png", _make_image_bytes(), "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"score", "threshold", "is_anomaly", "polygons", "overlay_url"}
    assert isinstance(data["score"], float)
    assert data["overlay_url"] is None or data["overlay_url"].startswith("/static/")

def test_predict_modes(client, monkeypatch):
    # Controlar salida de anomaly_map_and_score para probar umbral
    def fake_ams(gray, *_, **__):
        size = gray.shape[0]
        heat = np.zeros((size,size), np.float32)
        heat_norm = np.zeros_like(heat)
        hmin, hmax = 0.0, 1.0
        score = 0.5
        return heat, heat_norm, hmin, hmax, score

    monkeypatch.setattr(appmod, "anomaly_map_and_score", fake_ams, raising=True)
    monkeypatch.setattr(appmod, "THRESHOLD", 0.5, raising=False)

    files = {"file": ("x.png", _make_image_bytes(), "image/png")}

    # threshold base 0.5 => is_anomaly False (score > thr)
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    assert r.json()["is_anomaly"] is False

    # sensitive => thr=0.4 => True
    r = client.post("/predict?mode=sensitive", files=files)
    assert r.json()["is_anomaly"] is True

    # strict => thr=0.6 => False
    r = client.post("/predict?mode=strict", files=files)
    assert r.json()["is_anomaly"] is False

    # thr manual => 0.49 => True
    r = client.post("/predict?thr=0.49", files=files)
    assert r.json()["is_anomaly"] is True