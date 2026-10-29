"""Business logic layer.

Services orchestrate repositories and external clients. They receive every
collaborator through the constructor (constructor injection), which keeps the
pipeline testable without a database or a network connection. The package facade
re-exports every service so callers keep a single import path.
"""

from app.services.ai_service import OpenAIClient
from app.services.lore_ingestion_service import LoreIngestionService
from app.services.rag_service import RAGService

__all__ = ["LoreIngestionService", "OpenAIClient", "RAGService"]
