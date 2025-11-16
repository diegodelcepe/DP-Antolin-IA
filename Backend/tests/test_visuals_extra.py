import os, numpy as np, cv2
import main as appmod


def test_save_visuals_roi_border(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "STATIC_DIR", str(tmp_path/"static"), raising=False)
    (tmp_path/"static").mkdir(parents=True, exist_ok=True)

    img = np.full((64,64), 10, np.uint8)
    heat = np.zeros((64,64), np.float32)
    heat[20:40, 20:40] = 1.0

    # ROI: un cuadrado central
    roi = np.zeros((64,64), np.uint8); roi[16:48,16:48]=255

    res = appmod.save_visuals_and_polys(
        img_gray=img, heat_norm=heat, area_min=10, base_name="roi",
        thr_norm=0.5, roi_mask=roi
    )
    # acepta 7+ retornos
    ovp, hp, mp = res[0], res[1], res[2]
    for p in (ovp, hp, mp):
        assert os.path.exists(p)

def test_save_visuals_auto_thresh_no_positive(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "STATIC_DIR", str(tmp_path/"static"), raising=False)
    (tmp_path/"static").mkdir(parents=True, exist_ok=True)
    img = np.zeros((64,64), np.uint8)
    heat = np.zeros((64,64), np.float32)  # todo 0, entra rama sin thr_norm y sin >0
    res = appmod.save_visuals_and_polys(
        img_gray=img, heat_norm=heat, area_min=9999, base_name="auto",
        thr_norm=None, roi_mask=None
    )
    ovp, hp, mp = res[0], res[1], res[2]
    for p in (ovp, hp, mp):
        assert os.path.exists(p)

def test_save_visuals_area_min_filters(tmp_path, monkeypatch):
    monkeypatch.setattr(appmod, "STATIC_DIR", str(tmp_path/"static"), raising=False)
    (tmp_path/"static").mkdir(parents=True, exist_ok=True)
    img = np.zeros((64,64), np.uint8)
    heat = np.zeros((64,64), np.float32); heat[10:12,10:12]=1.0  # muy pequeño
    res = appmod.save_visuals_and_polys(
        img_gray=img, heat_norm=heat, area_min=1000, base_name="small",
        thr_norm=0.1, roi_mask=None
    )
    polys = res[3]
    assert isinstance(polys, list) and len(polys)==0  # filtrado por área
