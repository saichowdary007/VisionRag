import os
import tempfile
from fastapi.testclient import TestClient


def test_retriever_index_and_search_fallback():
    # Use a temporary data dir and ensure it's picked up at import time
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DATA_DIR"] = tmp
        os.environ["RETRIEVER_INDEX_DIR"] = os.path.join(tmp, "index")
        os.environ["BYALDI_USE_FALLBACK"] = "1"
        # Import after setting env so server binds to the tmp dir
        import importlib
        server = importlib.import_module("retriever.server")

        client = TestClient(server.app)

        # Index two fake image paths
        doc_id = "doc1"
        images = [
            "/tmp/invoice_page1.png",
            "/tmp/terms_page2.png",
        ]
        r = client.post("/index", json={"doc_id": doc_id, "images": images, "texts": ["invoice foo", "terms appear on this page"]})
        assert r.status_code == 200
        assert r.json().get("images_indexed") == len(images)

        # Query that should match page text in fallback
        r = client.post("/search", json={"query": "terms", "k": 5, "doc_id": doc_id})
        assert r.status_code == 200
        data = r.json()
        hits = data.get("hits", [])
        assert isinstance(hits, list)
        # Ensure at least one hit references the matching file
        assert any("terms_page2.png" in h.get("image_path", "") for h in hits)
