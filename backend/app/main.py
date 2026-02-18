from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.models import Base
from app.db.session import engine
from app.api.routes_auth import router as auth_router
from app.api.routes_infer import router as infer_router
from app.api.routes_history import router as history_router

# Cria tabelas (SQLite local) no startup
Base.metadata.create_all(bind=engine)

# Garante pastas
Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)

app = FastAPI(title=settings.APP_NAME)

# CORS travado para localhost:3000
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# Logging simples em JSON
@app.middleware("http")
async def json_logger(request: Request, call_next):
    t0 = time.time()
    try:
        response: Response = await call_next(request)
        status_code = response.status_code
    except Exception:
        status_code = 500
        raise
    finally:
        latency_ms = int((time.time() - t0) * 1000)
        log = {
            "method": request.method,
            "path": request.url.path,
            "status": status_code,
            "latency_ms": latency_ms,
        }
        print(json.dumps(log, ensure_ascii=False))
    return response

@app.get("/health")
def health():
    return {"ok": True, "env": settings.ENV}

app.include_router(auth_router)
app.include_router(infer_router)
app.include_router(history_router)
