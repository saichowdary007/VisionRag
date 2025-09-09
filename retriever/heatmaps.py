from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def _seed_from(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def generate_placeholder_heatmap(query: str, image_path: str, out_path: str) -> Optional[str]:
    """Generate a deterministic placeholder heatmap.

    Notes:
    - This is a stub standing in for true late-interaction MaxSim visualization.
    - It produces a 32x32 grayscale map seeded by (query+path), upsamples to the
      page size and saves as a red-tinted overlay PNG.
    - Returns the absolute output path, or None if generation fails.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        rng = np.random.default_rng(_seed_from(query + image_path))
        grid = rng.random((32, 32)).astype(np.float32)
        # Normalize and upsample
        grid = (grid - grid.min()) / max(1e-6, (grid.max() - grid.min()))
        heat = Image.fromarray((grid * 255).astype(np.uint8), mode="L").resize((w, h), Image.BILINEAR)
        # Convert to red heat overlay with alpha
        red = Image.new("RGBA", (w, h), (255, 0, 0, 0))
        alpha = heat.point(lambda p: int(p * 0.6))  # 0..153 alpha
        red.putalpha(alpha)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        red.save(out, format="PNG")
        return str(out.resolve())
    except Exception:
        return None

