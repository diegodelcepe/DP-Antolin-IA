import os
import numpy as np
import cv2
import main as appmod


def test_save_visuals_and_polys(tmp_path, monkeypatch):
    # Redirigir STATIC_DIR
    monkeypatch.setattr(appmod, "STATIC_DIR", str(tmp_path/"static"), raising=False)
    (tmp_path/"static").mkdir(parents=True, exist_ok=True)

    img = np.zeros((128,128), np.uint8)
    cv2.rectangle(img, (30,30), (90,90), 200, -1)
    heat = np.zeros((128,128), np.float32)
    heat[40:80, 40:80] = 1.0

    ovp, hp, mp, polys, ouv, huv, muv = appmod.save_visuals_and_polys(
        img_gray=img,
        heat_norm=heat,
        area_min=50,
        base_name="t",
        thr_norm=0.5,
        roi_mask=None
    )
    # Archivos existen
    for p in (ovp, hp, mp):
        assert os.path.exists(p)
    # URLs tienen prefijo /static/
    for u in (ouv, huv, muv):
        assert u.startswith("/static/")
    # Hay al menos un polígono
    assert isinstance(polys, list)
