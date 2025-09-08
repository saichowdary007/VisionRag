from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
import yaml


@dataclass
class Config:
    device: str = os.getenv("DEVICE", "mps")
    max_memory_gb: int = int(os.getenv("MAX_MEMORY_GB", "8"))
    db_path: str = os.getenv("DB_PATH", "./data/documents.db")
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "./data/vectors.faiss")
    # Primary dense text index (float)
    faiss_binary_index_path: str = os.getenv("FAISS_BINARY_INDEX_PATH", "./data/vectors.bin")
    faiss_binary_ids_path: str = os.getenv("FAISS_BINARY_IDS_PATH", "./data/vectors.bin.ids.npy")
    # Vision/image index (float)
    faiss_image_index_path: str = os.getenv("FAISS_IMAGE_INDEX_PATH", "./data/vision.faiss")
    docs_dir: str = os.getenv("DOCS_DIR", "./documents")
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "9108"))
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", os.path.expanduser("~/.cache/rag_models"))

    # models
    text_embedding_model: str = os.getenv("TEXT_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
    reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    # Support both SIGLIP_MODEL and VISION_EMBEDDING_MODEL envs
    siglip_model: str = os.getenv("SIGLIP_MODEL", os.getenv("VISION_EMBEDDING_MODEL", "google/siglip-base-patch16-224"))
    enable_vision: bool = os.getenv("ENABLE_VISION", "false").lower() in ("1", "true", "yes")
    enable_cross_modal: bool = os.getenv("ENABLE_CROSS_MODAL", "false").lower() in ("1", "true", "yes")

    # generation
    generator: str = os.getenv("GENERATOR", "extractive")  # extractive | lm_studio
    lm_studio_url: str = os.getenv("LM_STUDIO_URL", "")
    lm_studio_model: str = os.getenv("LM_STUDIO_MODEL", "google/gemma-3-12b")
    lm_studio_temperature: float = float(os.getenv("LM_STUDIO_TEMPERATURE", "0.1"))
    lm_studio_max_tokens: int = int(os.getenv("LM_STUDIO_MAX_TOKENS", "1024"))
    lm_studio_system_prompt: str = os.getenv("LM_STUDIO_SYSTEM_PROMPT", "")
    lm_studio_timeout: int = int(os.getenv("LM_STUDIO_TIMEOUT", "120"))
    lm_studio_max_retries: int = int(os.getenv("LM_STUDIO_MAX_RETRIES", "3"))

    # caching
    redis_url: str = os.getenv("REDIS_URL", "")
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    invalidate_on_reindex: bool = os.getenv("INVALIDATE_ON_REINDEX", "false").lower() in ("1", "true", "yes")

    # retrieval strategy
    use_binary_primary: bool = os.getenv("USE_BINARY_FILTER", "false").lower() in ("1", "true", "yes")
    binary_filter_candidates: int = int(os.getenv("BINARY_FILTER_CANDIDATES", "1000"))
    cross_modal_weight: float = float(os.getenv("CROSS_MODAL_WEIGHT", "0.3"))

    # query fusion (RAG-Fusion)
    enable_query_fusion: bool = os.getenv("ENABLE_QUERY_FUSION", "false").lower() in ("1", "true", "yes")
    query_expansions: int = int(os.getenv("QUERY_EXPANSIONS", "3"))
    query_fusion_rrf_k: int = int(os.getenv("QUERY_FUSION_RRF_K", "60"))
    query_expansion_temperature: float = float(os.getenv("QUERY_EXPANSION_TEMPERATURE", "0.2"))

    # retrieval candidates tuning
    rerank_candidates: int = int(os.getenv("RERANK_CANDIDATES", "50"))
    search_k_max: int = int(os.getenv("SEARCH_K_MAX", "100"))
    search_k_multiplier: int = int(os.getenv("SEARCH_K_MULTIPLIER", "3"))
    image_search_topk: int = int(os.getenv("IMAGE_SEARCH_TOPK", "60"))

    # vision reranker (optional CLIP)
    enable_vision_reranker: bool = os.getenv("ENABLE_VISION_RERANKER", "false").lower() in ("1", "true", "yes")
    vision_reranker_model: str = os.getenv("VISION_RERANKER_MODEL", "openai/clip-vit-base-patch32")
    vision_rerank_weight: float = float(os.getenv("VISION_RERANK_WEIGHT", "0.7"))
    vision_rerank_max_images: int = int(os.getenv("VISION_RERANK_MAX_IMAGES", "12"))

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        cfg = Config()
        p = Path(path)
        if p.exists():
            data = yaml.safe_load(p.read_text())
            env_map = {
                "device": "DEVICE",
                "max_memory_gb": "MAX_MEMORY_GB",
                "db_path": "DB_PATH",
                "faiss_index_path": "FAISS_INDEX_PATH",
                "faiss_image_index_path": "FAISS_IMAGE_INDEX_PATH",
                "docs_dir": "DOCS_DIR",
                "prometheus_port": "PROMETHEUS_PORT",
                "model_cache_dir": "MODEL_CACHE_DIR",
                "text_embedding_model": "TEXT_EMBEDDING_MODEL",
                "reranker_model": "RERANKER_MODEL",
                "siglip_model": "SIGLIP_MODEL",
                "enable_cross_modal": "ENABLE_CROSS_MODAL",
                "enable_vision": "ENABLE_VISION",
                "generator": "GENERATOR",
                "lm_studio_url": "LM_STUDIO_URL",
                "lm_studio_model": "LM_STUDIO_MODEL",
                "lm_studio_temperature": "LM_STUDIO_TEMPERATURE",
                "lm_studio_max_tokens": "LM_STUDIO_MAX_TOKENS",
                "lm_studio_system_prompt": "LM_STUDIO_SYSTEM_PROMPT",
                "lm_studio_timeout": "LM_STUDIO_TIMEOUT",
                "lm_studio_max_retries": "LM_STUDIO_MAX_RETRIES",
                "use_binary_primary": "USE_BINARY_FILTER",
                "binary_filter_candidates": "BINARY_FILTER_CANDIDATES",
                "cross_modal_weight": "CROSS_MODAL_WEIGHT",
                "invalidate_on_reindex": "INVALIDATE_ON_REINDEX",
                "enable_query_fusion": "ENABLE_QUERY_FUSION",
                "query_expansions": "QUERY_EXPANSIONS",
                "query_fusion_rrf_k": "QUERY_FUSION_RRF_K",
                "query_expansion_temperature": "QUERY_EXPANSION_TEMPERATURE",
                "rerank_candidates": "RERANK_CANDIDATES",
                "search_k_max": "SEARCH_K_MAX",
                "search_k_multiplier": "SEARCH_K_MULTIPLIER",
                "image_search_topk": "IMAGE_SEARCH_TOPK",
                "enable_vision_reranker": "ENABLE_VISION_RERANKER",
                "vision_reranker_model": "VISION_RERANKER_MODEL",
                "vision_rerank_weight": "VISION_RERANK_WEIGHT",
                "vision_rerank_max_images": "VISION_RERANK_MAX_IMAGES",
            }
            for k, v in (data or {}).items():
                if hasattr(cfg, k):
                    # Do not override explicit environment variables
                    env_key = env_map.get(k)
                    if env_key and os.getenv(env_key) not in (None, ""):
                        continue
                    setattr(cfg, k, v)
        # Normalize paths
        cfg.db_path = str(Path(cfg.db_path))
        cfg.faiss_index_path = str(Path(cfg.faiss_index_path))
        cfg.faiss_binary_index_path = str(Path(cfg.faiss_binary_index_path))
        cfg.faiss_binary_ids_path = str(Path(cfg.faiss_binary_ids_path))
        cfg.faiss_image_index_path = str(Path(cfg.faiss_image_index_path))
        cfg.docs_dir = str(Path(cfg.docs_dir))
        return cfg


def get_config() -> Config:
    # Try config file if mounted
    cfg_path = Path("./config/config.yaml")
    if cfg_path.exists():
        return Config.from_yaml(cfg_path)
    return Config()
