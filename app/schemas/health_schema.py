"""Схемы, описывающие эксплуатационные эндпоинты сервиса."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Результат пробы готовности."""

    status: str
    version: str
    vector_store: str
    """Доступность Qdrant: ``ok`` или ``unavailable``."""
