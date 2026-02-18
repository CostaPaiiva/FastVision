from __future__ import annotations

import json
from typing import List

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, get_db
from app.db.models import Result, User

router = APIRouter(prefix="/history", tags=["history"])


@router.get("")
def history(
    limit: int = Query(default=50, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    stmt = (
        select(Result)
        .where(Result.user_email == user.email)
        .order_by(desc(Result.created_at))
        .limit(limit)
    )
    rows = db.execute(stmt).scalars().all()

    out = []
    for r in rows:
        try:
            parsed = json.loads(r.result_json)
        except Exception:
            parsed = {"error": "invalid_json"}

        out.append(
            {
                "id": r.id,
                "created_at": r.created_at,
                "imgsz": r.imgsz,
                "conf": r.conf,
                "iou": r.iou,
                "model_name": r.model_name,
                "detections_count": len(parsed.get("detections", [])) if isinstance(parsed, dict) else 0,
            }
        )
    return {"items": out}
