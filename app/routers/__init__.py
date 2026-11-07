"""HTTP-слой: только маршрутизация, валидация запросов и перевод ошибок."""

from fastapi import APIRouter

from app.routers.admin import router as admin_router
from app.routers.archive import router as archive_router
from app.routers.auth import router as auth_router

router = APIRouter()
router.include_router(archive_router)
router.include_router(auth_router)
router.include_router(admin_router)

__all__ = ["router"]
