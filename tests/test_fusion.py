from __future__ import annotations

import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "rag-service"))


from rag_service.search.query_transform import expand_queries_lmstudio


class TestFusion(unittest.TestCase):
    def test_expand_queries_fallback(self):
        # Empty URL triggers exception path and returns [query]
        q = "replace coolant pump"
        out = expand_queries_lmstudio(url="", model="", query=q, n=3, timeout_s=1)
        self.assertEqual(out[0], q)
        self.assertGreaterEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
