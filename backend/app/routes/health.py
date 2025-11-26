from fastapi import APIRouter
from ..schemas import HealthCheck
from ..config import settings

router = APIRouter()

@router.get("/health", response_model=HealthCheck)
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0"
    }
