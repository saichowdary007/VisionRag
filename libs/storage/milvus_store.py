from __future__ import annotations

from typing import Any

try:
    from pymilvus import MilvusClient, DataType
except Exception:  # pragma: no cover - optional import for local dev
    MilvusClient = None  # type: ignore
    DataType = None  # type: ignore


def get_or_create_collection(client: Any, name: str, dim: int) -> None:
    if client is None:
        raise RuntimeError("Milvus client is not available")
    if not client.has_collection(collection_name=name):
        schema = client.create_schema(auto_id=True, description="ColPali multi-vector store")
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="page_id", datatype=DataType.VARCHAR, max_length=1024)
        index_params = client.prepare_index_params()
        # AUTOINDEX lets Milvus choose the best index for the data
        index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="IP")
        client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=index_params,
            consistency_level="Eventually",
        )


