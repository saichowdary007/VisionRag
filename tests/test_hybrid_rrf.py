import os
import tempfile
import importlib
from fastapi.testclient import TestClient


def test_hybrid_rrf_bm25_boosts_keyword_doc():
    # Force fallback so we can control texts
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["DATA_DIR"] = tmp
        os.environ["RETRIEVER_INDEX_DIR"] = os.path.join(tmp, "index")
        os.environ["BYALDI_USE_FALLBACK"] = "1"
        os.environ["HYBRID_ALPHA"] = "0.5"

        server = importlib.import_module("retriever.server")
        client = TestClient(server.app)

        doc_id = "doc1"
        images = [
            "/tmp/alpha_page1.png",
            "/tmp/bravo_page2.png",
        ]
        texts = ["", "specialkeyword is present here"]
        r = client.post("/index", json={"doc_id": doc_id, "images": images, "texts": texts})
        assert r.status_code == 200

        r = client.post("/search", json={"query": "specialkeyword", "k": 2, "doc_id": doc_id})
        assert r.status_code == 200
        hits = r.json().get("hits", [])
        assert len(hits) >= 1
        # Expect the page with the keyword to appear first
        top = hits[0]
        assert top["image_path"].endswith("bravo_page2.png")

