import numpy as np, cv2
import main as appmod


def _png_bytes(w=16, h=16):
    img = np.zeros((h, w, 3), np.uint8)
    cv2.circle(img, (w//2, h//2), min(w,h)//4, (0,0,255), -1)
    ok, buf = cv2.imencode(".png", img); assert ok
    return buf.tobytes()

def test_predict_savevis_disabled(client, monkeypatch):
    monkeypatch.setattr(appmod, "SAVE_VIS", False, raising=False)
    files = {"file": ("x.png", _png_bytes(), "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 200
    data = r.json()
    assert "overlay_url" in data and data["overlay_url"] is None

def test_predict_polygons_only_when_anomaly(client, monkeypatch):
    def fake_ams(gray, *_, **__):
        size = gray.shape[0]
        heat = np.zeros((size,size), np.float32)
        heat[4:-4, 4:-4] = 1.0
        return heat, heat, 0.0, 1.0, 0.1  # score bajo
    monkeypatch.setattr(appmod, "anomaly_map_and_score", fake_ams, raising=True)
    monkeypatch.setattr(appmod, "THRESHOLD", 0.5, raising=False)
    monkeypatch.setattr(appmod, "SAVE_VIS", True, raising=False)
    ok, buf = cv2.imencode(".png", np.zeros((16,16,3), np.uint8)); assert ok
    r = client.post("/predict", files={"file": ("x.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200
    data = r.json()
    assert data["is_anomaly"] is False
    assert data["polygons"] == []
