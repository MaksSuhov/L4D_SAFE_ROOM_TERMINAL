"""Векторный слой: адаптеры Qdrant за портами приложения.

Фасад пакета реэкспортирует адаптеры и клиент хранилища, поэтому точки сборки
импортируют всё векторное из одного места, а домен и сервисы о Qdrant не знают.
"""

from app.vector.bm25 import Bm25Encoder, SparseEncoderProtocol
from app.vector.character_repository import QdrantCharacterRepository
from app.vector.client import VectorStore
from app.vector.lore_repository import QdrantLoreChunkRepository

__all__ = [
    "Bm25Encoder",
    "QdrantCharacterRepository",
    "QdrantLoreChunkRepository",
    "SparseEncoderProtocol",
    "VectorStore",
]
