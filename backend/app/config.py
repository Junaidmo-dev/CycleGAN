import os
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    APP_NAME: str = "DeepClean AI"
    API_PREFIX: str = "/api"
    DEBUG: bool = False
    # Server configuration (matches .env)
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    # Redis configuration (matches .env)
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    @property
    def REDIS_URL(self) -> str:
        """Construct Redis URL from host, port and db for Celery usage."""
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    # Security
    SECRET_KEY: str = "super-secret-key-change-me"
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    MODEL_PATH: Path = BASE_DIR / "models" / "generator.pth"
    LOG_PATH: Path = BASE_DIR / "logs" / "app.log"
    STORAGE_PATH: Path = BASE_DIR / "storage"
    
    # Security
    ALLOWED_ORIGINS: list = ["*"]
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10 MB
    
    # Async Tasks
    ENABLE_BACKGROUND_TASKS: bool = True
    REDIS_URL: str = "redis://redis:6379/0"
    
    class Config:
        env_file = ".env"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.STORAGE_PATH, exist_ok=True)
os.makedirs(settings.LOG_PATH.parent, exist_ok=True)
os.makedirs(settings.MODEL_PATH.parent, exist_ok=True)
