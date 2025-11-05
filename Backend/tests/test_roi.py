import os
import cv2
import numpy as np
import main as appmod

def test_build_roi_mask_with_ignore(monkeypatch):
    monkeypatch.setattr(appmod, "IGNORE_BORDER_PCT", 10.0, raising=False)
    m = appmod.build_roi_mask(100)
    assert m.shape == (100, 100)
    # Bordes a cero
    assert m[0, 50] == 0 and m[-1, 50] == 0 and m[50, 0] == 0 and m[50, -1] == 0
    # Centro en ROI
    assert m[50, 50] == 255

def test_build_roi_mask_with_file(tmp_path, monkeypatch):
    mask = np.zeros((50,50), np.uint8)
    cv2.rectangle(mask, (10,10), (39,39), 255, -1)
    p = tmp_path/"roi.png"
    cv2.imwrite(str(p), mask)
    monkeypatch.setattr(appmod, "IGNORE_BORDER_PCT", 0.0, raising=False)
    monkeypatch.setattr(appmod, "ROI_PATH", str(p), raising=False)
    m = appmod.build_roi_mask(100)
    # Debe tener ROI centrada (tras resize)
    assert m.sum() > 0
