"""ORM models of the application.

The package facade re-exports every mapper, so the rest of the code keeps a
single stable import path (``from app.models import Character``) while the
definitions live in one module per aggregate.
"""

from app.models.character_model import Character
from app.models.lore_chunk_model import EMBEDDING_DIMENSIONS, LoreChunk

__all__ = ["EMBEDDING_DIMENSIONS", "Character", "LoreChunk"]
