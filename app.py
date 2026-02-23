# Habilita o uso de anotações de tipo de forma mais flexível (PEP 563)
from __future__ import annotations

# Importa o módulo 'os' para interagir com o sistema operacional (ex: manipulação de caminhos, criação de diretórios)
import os
# Importa o módulo 'time' para funções relacionadas a tempo (ex: obter timestamp atual)
import time
# Importa tipos específicos do módulo 'typing' para melhorar a legibilidade e verificação de tipos do código
from typing import Dict, List, Any, Optional

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from db import init_db, upsert_person, add_image, list_people, list_images, delete_person
from yolo_backend import YOLODetector, PredictConfig
from face_recog import get_face_cascade, preprocess_face, train_lbph, predict_faces
from exporters import export_csv, export_json

APP_TITLE = "CV App Local (YOLO + Reconhecimento por Nome)"
RUNS_DIR = "runs"
GALLERY_DIR = "gallery"

DEFAULT_YOLO_MODELS = ["yolo11n.pt", "yolo11s.pt"]  # no CPU, prefira yolo11n

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

@st.cache_resource
def load_yolo(weights: str) -> YOLODetector:
    return YOLODetector(weights)

def build_face_training_set() -> tuple[List[tuple[np.ndarray, int]], Dict[int, str]]:
    people = list_people()
    label_to_name = {pid: name for pid, name in people}

    cascade = get_face_cascade()
    samples: List[tuple[np.ndarray, int]] = []
    for _img_id, person_id, path in list_images():
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        if len(faces) == 0:
            continue
        x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
        face = preprocess_face(gray, (x, y, w, h))
        samples.append((face, int(person_id)))

    return samples, label_to_name

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Windows + CPU, tudo local. YOLO (objetos) + Reconhecimento por Nome (faces) + CSV/JSON por frame.")

init_db()
ensure_dir(RUNS_DIR)
ensure_dir(GALLERY_DIR)

# Sidebar
with st.sidebar:
    st.header("YOLO (Objetos)")
    yolo_weights = st.selectbox("Modelo", DEFAULT_YOLO_MODELS, index=0)
    conf = st.slider("Confiança", 0.01, 0.99, 0.25, 0.01)
    iou = st.slider("IoU", 0.10, 0.90, 0.45, 0.01)
    imgsz = st.select_slider("imgsz (CPU)", options=[320, 416, 480, 640], value=480)
    max_det = st.slider("max_det", 50, 500, 200, 50)

    st.divider()
    st.header("Reconhecimento (Faces)")
    face_threshold = st.slider("Threshold LBPH (menor = mais exigente)", 30.0, 120.0, 70.0, 1.0)

    st.divider()
    st.header("Export")
    export_format = st.selectbox("Formato", ["CSV", "JSON"], index=0)
    export_every_n = st.slider("Salvar registro a cada N frames", 1, 30, 1, 1)

cfg = PredictConfig(conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device="cpu")
yolo = load_yolo(yolo_weights)

# Haar cascade
try:
    cascade = get_face_cascade()
except Exception as e:
    cascade = None
    st.sidebar.error("Haar cascade não encontrado. Veja o PDF/README para corrigir.")
    st.sidebar.caption(str(e))

tabA, tabB = st.tabs(["Cadastro (Banco de imagens + nomes)", "Live (Webcam / RTSP + Export)"])

# Cadastro
with tabA:
    st.subheader("Cadastro de pessoas (imagens + nome) — SQLite + pasta gallery/")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        name = st.text_input("Nome da pessoa")
        imgs = st.file_uploader(
            "Envie 1 ou mais imagens (jpg/png) com o rosto visível",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if st.button("Salvar no banco", disabled=not (name and imgs)):
            if cascade is None:
                st.error("Sem Haar cascade. Corrija o arquivo assets/haarcascade_frontalface_default.xml e tente de novo.")
            else:
                pid = upsert_person(name.strip())
                person_dir = os.path.join(GALLERY_DIR, f"{pid}_{name.strip().replace(' ', '_')}")
                ensure_dir(person_dir)

                saved = 0
                skipped = 0
                for f in imgs:
                    pil = Image.open(f).convert("RGB")
                    rgb = np.array(pil)
                    bgr = rgb_to_bgr(rgb)

                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
                    if len(faces) == 0:
                        skipped += 1
                        continue

                    ts = int(time.time() * 1000)
                    out_path = os.path.join(person_dir, f"{ts}.jpg")
                    cv2.imwrite(out_path, bgr)
                    add_image(pid, out_path)
                    saved += 1

                st.success(f"Salvas {saved} imagens para '{name}'.")
                if skipped:
                    st.info(f"{skipped} imagens foram ignoradas (nenhuma face detectada).")

    with col2:
        st.markdown("### Pessoas cadastradas")
        people = list_people()
        if not people:
            st.info("Nenhuma pessoa cadastrada ainda.")
        else:
            for pid, pname in people:
                c1, c2 = st.columns([3, 1])
                c1.write(f"**{pname}** (id={pid})")
                if c2.button("Excluir", key=f"del_{pid}"):
                    delete_person(pid)
                    st.warning(f"Removido: {pname}")
                    st.rerun()

        st.divider()
        st.markdown("### Treinar reconhecedor")
        if st.button("Treinar/Atualizar modelo de reconhecimento"):
            if cascade is None:
                st.error("Sem Haar cascade. Corrija o arquivo assets/haarcascade_frontalface_default.xml.")
            else:
                samples, label_to_name = build_face_training_set()
                recognizer = train_lbph(samples)
                if recognizer is None:
                    st.error("Poucas amostras. Cadastre mais imagens (mínimo ~2) com faces detectáveis.")
                else:
                    st.session_state["recognizer"] = recognizer
                    st.session_state["label_to_name"] = label_to_name
                    st.success(f"Treinado! Amostras: {len(samples)} | Pessoas: {len(label_to_name)}")

# Live
with tabB:
    st.subheader("Live: Webcam ou RTSP (OpenCV) + export por frame")
    st.caption("Dica CPU: use yolo11n + imgsz 416/480 para ficar leve.")

    source_type = st.radio("Fonte", ["Webcam", "RTSP"], horizontal=True)
    cam_index = st.number_input("Indice da webcam", 0, 10, 0, 1, disabled=(source_type != "Webcam"))
    rtsp_url = st.text_input("RTSP URL (ex: rtsp://user:pass@ip:554/...) ", disabled=(source_type != "RTSP"))

    colL, colR = st.columns([2, 1], gap="large")

    with colR:
        st.markdown("### Controles")
        start = st.toggle("Iniciar", value=False)
        save_annotated_video = st.checkbox("Gravar video anotado (mp4)", value=False)
        max_seconds = st.slider("Limite (segundos) para rodar", 5, 300, 30, 5)
        run_name = st.text_input("Nome da execucao", value=time.strftime("%Y%m%d-%H%M%S"))

        st.markdown("### Reconhecimento")
        use_face = st.checkbox("Ativar reconhecimento por nome", value=True)
        if use_face and "recognizer" not in st.session_state:
            st.warning("Treine o modelo na aba Cadastro antes de usar reconhecimento.")

    with colL:
        frame_box = st.empty()
        info_box = st.empty()

    if start:
        if cascade is None:
            st.error("Sem Haar cascade. Corrija o arquivo assets/haarcascade_frontalface_default.xml.")
        else:
            src = int(cam_index) if source_type == "Webcam" else rtsp_url.strip()
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                st.error("Nao consegui abrir a fonte. Verifique webcam/RTSP e permissoes.")
            else:
                records: List[Dict[str, Any]] = []
                out_dir = os.path.join(RUNS_DIR, run_name)
                ensure_dir(out_dir)

                writer = None
                out_video = None
                if save_annotated_video:
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                    out_video = os.path.join(out_dir, "annotated.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(out_video, fourcc, float(fps), (w, h))

                t0 = time.time()
                frame_idx = 0

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        info_box.warning("Falha ao ler frame. Encerrando.")
                        break

                    annotated, dets = yolo.predict(frame, cfg)

                    faces_out = []
                    if use_face and "recognizer" in st.session_state:
                        faces_out = predict_faces(
                            frame_bgr=frame,
                            recognizer=st.session_state["recognizer"],
                            label_to_name=st.session_state.get("label_to_name", {}),
                            cascade=cascade,
                            threshold=face_threshold
                        )
                        for fname, dist, (x, y, w, h) in faces_out:
                            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            cv2.putText(
                                annotated,
                                f"{fname} ({dist:.0f})",
                                (x, max(0, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2
                            )

                    if frame_idx % export_every_n == 0:
                        ts = time.time()
                        for d in dets:
                            records.append({
                                "run": run_name,
                                "frame_idx": frame_idx,
                                "timestamp": ts,
                                "type": "yolo",
                                **d
                            })
                        for fname, dist, (x, y, w, h) in faces_out:
                            records.append({
                                "run": run_name,
                                "frame_idx": frame_idx,
                                "timestamp": ts,
                                "type": "face",
                                "name": fname,
                                "distance": float(dist),
                                "x1": int(x), "y1": int(y), "x2": int(x + w), "y2": int(y + h)
                            })

                    if writer is not None:
                        writer.write(annotated)

                    frame_box.image(bgr_to_rgb(annotated), use_container_width=True)
                    info_box.info(f"Frame: {frame_idx} | YOLO: {len(dets)} | Faces: {len(faces_out)}")

                    frame_idx += 1
                    if time.time() - t0 >= max_seconds:
                        break

                cap.release()
                if writer is not None:
                    writer.release()

                if export_format == "CSV":
                    out_path = os.path.join(out_dir, "detections.csv")
                    export_csv(records, out_path)
                else:
                    out_path = os.path.join(out_dir, "detections.json")
                    export_json(records, out_path)

                st.success(f"Execucao salva em: {out_dir}")
                st.write("Arquivo export:", out_path)
                if out_video and os.path.exists(out_video):
                    st.video(out_video)
