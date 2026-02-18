from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "skillup-yolo-local"
    ENV: str = "local"

    JWT_SECRET: str = "change-me"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    DATABASE_URL: str = "sqlite:///./app.db"

    YOLO_WEIGHTS: str = "./weights/yolov8n.pt"
    DEFAULT_IMGSZ: int = 640
    DEFAULT_CONF: float = 0.25
    DEFAULT_IOU: float = 0.7

    TORCH_THREADS: int = 4

    DATA_DIR: str = "./data"
    UPLOADS_DIR: str = "./data/uploads"
    OUTPUTS_DIR: str = "./data/outputs"

    CORS_ORIGINS: str = "http://localhost:3000"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
