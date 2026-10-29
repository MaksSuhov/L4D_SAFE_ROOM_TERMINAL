"""Schemas describing the operational endpoints of the service."""

from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Result of the readiness probe."""

    status: str
    version: str
    database: str
