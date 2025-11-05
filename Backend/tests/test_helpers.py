import numpy as np
import torch
import main as appmod

def test_bgr_to_gray_256(sample_bgr):
    g = appmod.bgr_to_gray_256(sample_bgr, 256)
    assert g.shape == (256, 256)
    assert g.dtype == np.uint8

def test_to_tensor_3ch(sample_bgr):
    g = appmod.bgr_to_gray_256(sample_bgr, 128)
    x = appmod.to_tensor_3ch(g)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (1, 3, 128, 128)
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0

def test_patchify_feature_map_stride():
    fmap = torch.randn(24, 10, 10)  # (C,H,W)
    p1 = appmod.patchify_feature_map(fmap, stride=1)
    p2 = appmod.patchify_feature_map(fmap, stride=2)
    assert p1.shape[0] == 10*10 and p1.shape[1] == 24
    assert p2.shape[0] == 5*5 and p2.shape[1] == 24

def test_l2_normalize_rows():
    X = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    Y = appmod.l2_normalize_rows(X)
    norms = np.linalg.norm(Y, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)