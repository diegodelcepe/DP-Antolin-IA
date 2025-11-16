import os
import main as appmod


def test_root_without_index(monkeypatch):
    # Simula que no existe templates/index.html
    monkeypatch.setattr(appmod, "BASE_DIR", os.path.join(os.getcwd(), "no_such_dir"), raising=False)
    r = appmod.app.router.routes  # fuerza carga de rutas
    from fastapi.testclient import TestClient
    c = TestClient(appmod.app)
    resp = c.get("/")
    assert resp.status_code == 200
    j = resp.json()
    assert isinstance(j, dict) and "templates/index.html no encontrado" in j.get("detail","")

def test_startup_no_memory_bank(monkeypatch):
    # Mock para que falle load_knn pero la app siga funcionando
    monkeypatch.setattr(appmod, "load_knn", lambda *a, **k: appmod.NearestNeighbors(n_neighbors=1), raising=True)
    # re-ejecuta startup de forma segura
    appmod.BACKBONE, appmod.HOOK2, appmod.HOOK3 = appmod.build_backbone()
    appmod.KNN = appmod.load_knn("", 1)
    assert appmod.BACKBONE is not None
