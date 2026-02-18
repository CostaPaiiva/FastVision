from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import String, Text, Float, Integer, DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), primary_key=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    role: Mapped[str] = mapped_column(String(32), default="user", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)


class Result(Base):
    __tablename__ = "results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, index=True)  # uuid
    user_email: Mapped[str] = mapped_column(String(320), index=True, nullable=False)

    input_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    output_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)

    imgsz: Mapped[int] = mapped_column(Integer, nullable=False)
    conf: Mapped[float] = mapped_column(Float, nullable=False)
    iou: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
