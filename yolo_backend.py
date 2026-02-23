from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from ultralytics import YOLO

@dataclass
class PredictConfig:
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    max_det: int = 300
    classes: Optional[List[int]] = None
    device: str = "cpu"
    half: bool = False
    agnostic_nms: bool = False

class YOLODetector:
    def __init__(self, weights_path: str):
        self.model = YOLO(weights_path)

    def predict(self, frame_bgr: np.ndarray, cfg: PredictConfig) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Retorna:
          - frame anotado (BGR)
          - lista de detecções (uma por box)
        """
        results = self.model.predict(
            source=frame_bgr,
            conf=cfg.conf,
            iou=cfg.iou,
            imgsz=cfg.imgsz,
            max_det=cfg.max_det,
            classes=cfg.classes,
            device=cfg.device,
            half=cfg.half,
            agnostic_nms=cfg.agnostic_nms,
            verbose=False
        )
        r = results[0]
        annotated = r.plot()

        dets: List[Dict[str, Any]] = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                dets.append({
                    "class_id": int(k),
                    "class_name": str(names[int(k)]),
                    "confidence": float(c),
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                })

        return annotated, dets
