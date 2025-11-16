import io, os, numpy as np
import main as appmod


def test_predict_empty_file(client):
    files = {"file": ("empty.png", b"", "image/png")}
    r = client.post("/predict", files=files)
    assert r.status_code == 400
    assert "Archivo vacío" in r.text or "Archivo vacío" in str(r.json())

def test_predict_invalid_image(client):
    files = {"file": ("bad.bin", b"\x00\x01\x02\x03", "application/octet-stream")}
    r = client.post("/predict", files=files)
    assert r.status_code == 400
    assert "No se pudo decodificar" in r.text or "decodificar" in str(r.json())

def test_predict_thr_not_number(client):
    files = {"file": ("x.png", np.zeros((4,4,3), dtype=np.uint8).tobytes(), "image/png")}
    # Fuerza imagen válida: mejor usa bytes reales de PNG
    import cv2
    ok, buf = cv2.imencode(".png", np.zeros((8,8,3), np.uint8))
    assert ok
    r = client.post("/predict?thr=hola", files={"file": ("x.png", buf.tobytes(), "image/png")})
    # FastAPI debería responder 422 (validation error) o 400 si tu código lo gestiona
    assert r.status_code in (400, 422)

def test_predict_unknown_mode(client):
    import cv2, numpy as np
    ok, buf = cv2.imencode(".png", np.zeros((8,8,3), np.uint8))
    assert ok
    r = client.post("/predict?mode=superstrict", files={"file": ("x.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200  # no debe crashear
    data = r.json()
    assert "score" in data and "threshold" in data
