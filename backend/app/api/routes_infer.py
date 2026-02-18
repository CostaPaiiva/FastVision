from __future__ import annotations

import mimetypes
import os
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.core.config import settings
from app.db.models import Result, User
from app.services.yolo_service import get_yolo_service

router = APIRouter(prefix="/infer", tags=["infer"])

MAX_BYTES = 10 * 1024 * 1024  # 10MB
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}


class InferOut(BaseModel):
    id: str
    model_name: str
    imgsz: int
    conf: float
    iou: float
    detections: list


@router.post("/image", response_model=InferOut)
async def infer_image(
    file: UploadFile = File(...),
    imgsz: int = Form(default=settings.DEFAULT_IMGSZ),
    conf: float = Form(default=settings.DEFAULT_CONF),
    iou: float = Form(default=settings.DEFAULT_IOU),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if imgsz not in (512, 640) and (imgsz < 320 or imgsz > 1280):
        raise HTTPException(status_code=400, detail="imgsz inválido (use 512 ou 640 por padrão)")

    if conf <= 0 or conf > 1:
        raise HTTPException(status_code=400, detail="conf inválido (0..1]")
    if iou <= 0 or iou > 1:
        raise HTTPException(status_code=400, detail="iou inválido (0..1]")

    if not file.content_type or file.content_type.lower() not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail="Tipo inválido. Aceito: JPEG/PNG/WEBP")

    # Lê bytes e valida tamanho (leve e simples)
    b = await file.read()
    if len(b) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="Arquivo > 10MB")

    # Prefixo único
    rid = str(uuid.uuid4())
    prefix = f"{rid}"

    # Executa YOLO (singleton)
    svc = get_yolo_service()
    try:
        out = svc.run_inference(
            image_bytes=b,
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou),
            save_prefix=prefix,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="Falha na inferência")

    # Persiste no DB
    rec = Result(
        id=rid,
        user_email=user.email,
        input_path=out["input_path"],
        output_path=out["output_path"],
        result_json=out["result_json"],
        imgsz=int(imgsz),
        conf=float(conf),
        iou=float(iou),
        model_name=svc.model_name,
    )
    db.add(rec)
    db.commit()

    return InferOut(
        id=rid,
        model_name=svc.model_name,
        imgsz=int(imgsz),
        conf=float(conf),
        iou=float(iou),
        detections=out["detections"],
    )


@router.get("/result/{id}")
def get_result(
    id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rec = db.get(Result, id)
    if not rec or rec.user_email != user.email:
        raise HTTPException(status_code=404, detail="Not found")
    return {
        "id": rec.id,
        "user_email": rec.user_email,
        "input_path": rec.input_path,
        "output_path": rec.output_path,
        "result_json": rec.result_json,
        "imgsz": rec.imgsz,
        "conf": rec.conf,
        "iou": rec.iou,
        "model_name": rec.model_name,
        "created_at": rec.created_at,
    }


@router.get("/result/{id}/image")
def get_result_image(
    id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rec = db.get(Result, id)
    if not rec or rec.user_email != user.email:
        raise HTTPException(status_code=404, detail="Not found")

    path = rec.output_path
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(path, media_type="image/png", filename=f"{id}.png")
