import os
from functools import lru_cache
class Settings:
    # Directories
    DATA_DIR: str = os.getenv("DATA_DIR", "/data")

    # Retriever service
    RETRIEVER_BASE_URL: str = os.getenv("RETRIEVER_BASE_URL", "http://retriever:8081")
    TOP_K: int = int(os.getenv("TOP_K", "5"))

    # API base URLs
    # Used to construct image URLs accessible to the VLM (host LM Studio)
    API_INTERNAL_BASE: str = os.getenv("API_INTERNAL_BASE", "http://host.docker.internal:8080")
    # Used by the browser
    API_PUBLIC_BASE: str = os.getenv("API_PUBLIC_BASE", "http://localhost:8080")
    # CORS
    WEB_ORIGIN: str = os.getenv("WEB_ORIGIN", "http://localhost:5173")

    # VLM (OpenAI-compatible). Support new VLM_* names with backwards LMSTUDIO_* aliases.
    VLM_BASE_URL: str = os.getenv("VLM_BASE_URL") or os.getenv("LMSTUDIO_BASE_URL", "http://host.docker.internal:1234")
    VLM_API_KEY: str = os.getenv("VLM_API_KEY") or os.getenv("LMSTUDIO_API_KEY", "lm-studio")
    VLM_MODEL: str = os.getenv("VLM_MODEL") or os.getenv("LMSTUDIO_MODEL", "gemma-3-4b-it")
    VLM_MAX_IMAGES: int = int(os.getenv("VLM_MAX_IMAGES", "5"))

    # Hybrid retrieval weight; applied inside retriever service, but kept here for UI hints
    HYBRID_ALPHA: float = float(os.getenv("HYBRID_ALPHA", "0.0"))

    # Rendering sizes
    PAGE_MAX_DIM: int = int(os.getenv("PAGE_MAX_DIM", "1024"))
    VLM_SQUARE: int = int(os.getenv("VLM_SQUARE", "896"))
@lru_cache()
def get_settings() -> Settings:
    return Settings()

