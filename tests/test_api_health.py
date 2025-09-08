import os
import tempfile
import importlib
from fastapi.testclient import TestClient


def test_healthz_ok():
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DATA_DIR"] = tmp
        # Import/reload after setting DATA_DIR so app writes under tmp
        from api import main
        importlib.reload(main)

        client = TestClient(main.app)
        r = client.get("/healthz")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, dict)
        assert data.get("ok") is True
