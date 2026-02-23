import os
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

CASCADE_PATH = os.path.join("assets", "haarcascade_frontalface_default.xml")

def ensure_assets() -> None:
    os.makedirs("assets", exist_ok=True)
    if not os.path.exists(CASCADE_PATH):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {CASCADE_PATH}\n"
            "Dica: copie o haarcascade_frontalface_default.xml do OpenCV para a pasta assets/."
        )

def get_face_cascade() -> cv2.CascadeClassifier:
    ensure_assets()
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if cascade.empty():
        raise RuntimeError("Falha ao carregar Haar Cascade. Verifique o arquivo XML.")
    return cascade

def detect_faces_gray(gray: np.ndarray, cascade: cv2.CascadeClassifier) -> List[Tuple[int, int, int, int]]:
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    return list(faces)

def preprocess_face(gray: np.ndarray, box: Tuple[int, int, int, int], size=(160, 160)) -> np.ndarray:
    x, y, w, h = box
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, size, interpolation=cv2.INTER_AREA)
    return face

def train_lbph(samples: List[Tuple[np.ndarray, int]]) -> Optional["cv2.face_LBPHFaceRecognizer"]:
    """
    samples: lista de (face_gray_160x160, label_int)
    """
    if len(samples) < 2:
        return None
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
    X = [s[0] for s in samples]
    y = np.array([s[1] for s in samples], dtype=np.int32)
    recognizer.train(X, y)
    return recognizer

def predict_faces(
    frame_bgr: np.ndarray,
    recognizer,
    label_to_name: Dict[int, str],
    cascade: cv2.CascadeClassifier,
    threshold: float = 70.0
):
    """
    threshold: quanto menor, mais exigente (50-90 costuma ser OK).
    Retorna lista de (name, distance, (x,y,w,h))
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = detect_faces_gray(gray, cascade)
    out = []
    for box in faces:
        face = preprocess_face(gray, box)
        label, dist = recognizer.predict(face)  # dist menor = melhor
        if dist <= threshold and label in label_to_name:
            out.append((label_to_name[label], float(dist), box))
        else:
            out.append(("desconhecido", float(dist), box))
    return out
