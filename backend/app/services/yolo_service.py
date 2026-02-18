from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from app.core.config import settings


@dataclass
class Detection:
    cls: int
    name: str
    conf: float
    box: Tuple[float, float, float, float]  # x1,y1,x2,y2


def _ensure_dirs():
    Path(settings.UPLOADS_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)


def _load_font():
    # Leve: tenta fonte padrão; se não existir, usa fallback do PIL.
    try:
        return ImageFont.load_default()
    except Exception:
        return None


class YoloService:
    def __init__(self):
        _ensure_dirs()
        self._model = None
        self._model_name = None
        self._init_model()

    def _init_model(self):
        # Limita threads do Torch (leve)
        try:
            import torch
            torch.set_num_threads(max(1, int(settings.TORCH_THREADS)))
        except Exception:
            pass

        weights_path = settings.YOLO_WEIGHTS
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"YOLO weights não encontrado em '{weights_path}'. "
                "Coloque yolov8n.pt em backend/weights/ ou rode scripts/download_weights.py"
            )

        from ultralytics import YOLO
        self._model = YOLO(weights_path)
        self._model_name = Path(weights_path).name

    @property
    def model_name(self) -> str:
        return self._model_name or "unknown"

    def run_inference(
        self,
        image_bytes: bytes,
        imgsz: int,
        conf: float,
        iou: float,
        save_prefix: str,
    ) -> Dict[str, Any]:
        """
        Retorna dict com:
        - id
        - input_path
        - output_path
        - result_json (string)
        - detections (lista)
        """
        _ensure_dirs()

        # Carrega imagem (corrige EXIF)
        img = Image.open(self._bytes_to_tempfile(image_bytes))
        img = ImageOps.exif_transpose(img).convert("RGB")

        # Salva input
        input_name = f"{save_prefix}_input.jpg"
        input_path = str(Path(settings.UPLOADS_DIR) / input_name)
        img.save(input_path, format="JPEG", quality=95)

        # Inferência
        # Ultralytics aceita numpy array
        arr = np.array(img)
        t0 = time.time()
        results = self._model.predict(
            source=arr,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
            device="cpu",
        )
        _ = time.time() - t0

        r0 = results[0]
        names = r0.names or {}

        dets: List[Detection] = []
        if r0.boxes is not None and len(r0.boxes) > 0:
            boxes_xyxy = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy()
            clss = r0.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(boxes_xyxy, confs, clss):
                dets.append(
                    Detection(
                        cls=int(k),
                        name=str(names.get(int(k), str(k))),
                        conf=float(c),
                        box=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )

        # Desenha output
        out_img = img.copy().convert("RGBA")
        draw = ImageDraw.Draw(out_img)
        font = _load_font()

        for d in dets:
            x1, y1, x2, y2 = d.box
            # Retângulo
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=3)
            label = f"{d.name} {d.conf:.2f}"
            # Fundo do label
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            pad = 3
            draw.rectangle([x1, max(0, y1 - th - 2 * pad), x1 + tw + 2 * pad, y1], fill=(0, 255, 0, 180))
            draw.text((x1 + pad, max(0, y1 - th - pad)), label, fill=(0, 0, 0, 255), font=font)

        output_name = f"{save_prefix}_output.png"
        output_path = str(Path(settings.OUTPUTS_DIR) / output_name)
        out_img.save(output_path, format="PNG")

        # JSON
        result = {
            "model": self.model_name,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "detections": [
                {"cls": d.cls, "name": d.name, "conf": d.conf, "box": list(d.box)}
                for d in dets
            ],
        }
        return {
            "input_path": input_path,
            "output_path": output_path,
            "result_json": json.dumps(result, ensure_ascii=False),
            "detections": result["detections"],
        }

    def _bytes_to_tempfile(self, b: bytes):
        # PIL precisa de file-like; usamos BytesIO (leve)
        from io import BytesIO
        return BytesIO(b)


# Singleton
_service: YoloService | None = None

def get_yolo_service() -> YoloService:
    global _service
    if _service is None:
        _service = YoloService()
    return _service
