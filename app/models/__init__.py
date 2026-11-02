"""ORM-модели приложения.

Фасад пакета реэкспортирует каждый маппер, поэтому остальной код сохраняет
единый стабильный путь импорта (``from app.models import Character``), пока
определения живут по одному модулю на агрегат.
"""

from app.models.character_model import Character
from app.models.lore_chunk_model import EMBEDDING_DIMENSIONS, LoreChunk

__all__ = ["EMBEDDING_DIMENSIONS", "Character", "LoreChunk"]
