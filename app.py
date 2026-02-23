# Habilita o uso de anotações de tipo de forma mais flexível (PEP 563)
from __future__ import annotations

# Importa o módulo 'os' para interagir com o sistema operacional (ex: manipulação de caminhos, criação de diretórios)
import os
# Importa o módulo 'time' para funções relacionadas a tempo (ex: obter timestamp atual)
import time
# Importa tipos específicos do módulo 'typing' para melhorar a legibilidade e verificação de tipos do código
from typing import Dict, List, Any, Optional

# Importa o módulo OpenCV para manipulação de imagens e visão computacional
import cv2
# Importa o módulo NumPy para operações com arrays numéricos, essencial para imagens
import numpy as np
# Importa o Streamlit, um framework para construir aplicativos web interativos em Python
import streamlit as st
# Importa a classe Image do módulo PIL (Pillow) para manipulação de imagens
from PIL import Image

# Importa funções específicas do módulo 'db' para interagir com o banco de dados
from db import init_db, upsert_person, add_image, list_people, list_images, delete_person
# Importa classes e configurações do módulo 'yolo_backend' para detecção de objetos YOLO
from yolo_backend import YOLODetector, PredictConfig
# Importa funções do módulo 'face_recog' para reconhecimento facial (cascade, preprocessamento, treinamento, predição)
from face_recog import get_face_cascade, preprocess_face, train_lbph, predict_faces
# Importa funções do módulo 'exporters' para exportar dados em formatos CSV e JSON
from exporters import export_csv, export_json

# Define uma constante para o título da aplicação
APP_TITLE = "CV App Local (YOLO + Reconhecimento por Nome)"
# Define o nome do diretório onde as execuções (runs) serão salvas
RUNS_DIR = "runs"
# Define o nome do diretório onde as imagens da galeria (cadastro de faces) serão armazenadas
GALLERY_DIR = "gallery"

# Define uma lista de modelos YOLO padrão para seleção, com preferência para o 'n' em CPU
DEFAULT_YOLO_MODELS = ["yolo11n.pt", "yolo11s.pt"]  # no CPU, prefira yolo11n

# Define uma função para garantir que um diretório exista, criando-o se necessário
def ensure_dir(path: str) -> None:
    # Cria o diretório no caminho especificado se ele não existir (exist_ok=True evita erro se já existir)
    os.makedirs(path, exist_ok=True)

# Define uma função para converter uma imagem do formato BGR (OpenCV) para RGB (Streamlit/PIL)
def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    # Usa cv2.cvtColor para a conversão de cores
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Define uma função para converter uma imagem do formato RGB para BGR
def rgb_to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    # Usa cv2.cvtColor para a conversão de cores
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# Decorador do Streamlit para armazenar em cache o recurso (YOLO Detector) para evitar recargas desnecessárias
@st.cache_resource
# Define uma função para carregar o modelo YOLO
def load_yolo(weights: str) -> YOLODetector:
    # Retorna uma instância de YOLODetector com os pesos especificados
    return YOLODetector(weights)

# Define uma função para construir o conjunto de dados de treinamento para o reconhecimento facial
def build_face_training_set() -> tuple[List[tuple[np.ndarray, int]], Dict[int, str]]:
    # Obtém a lista de pessoas cadastradas no banco de dados
    people = list_people()
    # Cria um dicionário que mapeia IDs de pessoas para seus nomes
    label_to_name = {pid: name for pid, name in people}

    # Carrega o classificador Haar cascade para detecção de faces
    cascade = get_face_cascade()
    # Inicializa uma lista vazia para armazenar as amostras de faces e seus rótulos
    samples: List[tuple[np.ndarray, int]] = []
    # Itera sobre todas as imagens cadastradas no banco de dados
    for _img_id, person_id, path in list_images():
        # Verifica se o arquivo da imagem existe no sistema de arquivos
        if not os.path.exists(path):
            # Se não existir, pula para a próxima imagem
            continue
        # Lê a imagem do caminho especificado
        img = cv2.imread(path)
        # Verifica se a imagem foi lida com sucesso
        if img is None:
            # Se não foi, pula para a próxima imagem
            continue
        # Converte a imagem para escala de cinza, necessário para a detecção de faces com Haar cascade
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detecta faces na imagem em escala de cinza usando o classificador Haar cascade
        faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
        # Verifica se alguma face foi detectada
        if len(faces) == 0:
            # Se nenhuma face foi detectada, pula para a próxima imagem
            continue
        # Seleciona a maior face detectada (ordena por área em ordem decrescente)
        x, y, w, h = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
        # Pré-processa a região da face para ser usada no treinamento do reconhecedor
        face = preprocess_face(gray, (x, y, w, h))
        # Adiciona a face pré-processada e o ID da pessoa à lista de amostras
        samples.append((face, int(person_id)))

    # Retorna as amostras de faces e o mapeamento de rótulos para nomes
    return samples, label_to_name

# Configura a página do Streamlit: define o título e o layout da página como "wide" (largo)
st.set_page_config(page_title=APP_TITLE, layout="wide")
# Exibe o título principal da aplicação na página
st.title(APP_TITLE)
# Exibe uma pequena descrição ou subtítulo da aplicação
st.caption("Windows + CPU, tudo local. YOLO (objetos) + Reconhecimento por Nome (faces) + CSV/JSON por frame.")

# Inicializa o banco de dados (SQLite)
init_db()
# Garante que o diretório para salvar as execuções (runs) exista
ensure_dir(RUNS_DIR)
# Garante que o diretório da galeria de imagens exista
ensure_dir(GALLERY_DIR)

# Inicia a barra lateral (sidebar) do Streamlit
with st.sidebar:
    # Exibe um cabeçalho para a seção de configurações do YOLO
    st.header("YOLO (Objetos)")
    # Cria um seletor para escolher o modelo YOLO, com o primeiro modelo como padrão
    yolo_weights = st.selectbox("Modelo", DEFAULT_YOLO_MODELS, index=0)
    # Cria um slider para ajustar o limiar de confiança (confidence threshold) para detecções YOLO
    conf = st.slider("Confiança", 0.01, 0.99, 0.25, 0.01)
    # Cria um slider para ajustar o limiar de IoU (Intersection over Union) para NMS (Non-Maximum Suppression)
    iou = st.slider("IoU", 0.10, 0.90, 0.45, 0.01)
    # Cria um slider para selecionar o tamanho da imagem de entrada (imgsz) para o YOLO, otimizado para CPU
    imgsz = st.select_slider("imgsz (CPU)", options=[320, 416, 480, 640], value=480)
    # Cria um slider para ajustar o número máximo de detecções (max_det)
    max_det = st.slider("max_det", 50, 500, 200, 50)

    # Adiciona um divisor visual na barra lateral
    st.divider()
    # Exibe um cabeçalho para a seção de reconhecimento facial
    st.header("Reconhecimento (Faces)")
    # Cria um slider para ajustar o limiar do reconhecedor LBPH (menor valor = mais exigente)
    face_threshold = st.slider("Threshold LBPH (menor = mais exigente)", 30.0, 120.0, 70.0, 1.0)

    # Adiciona um divisor visual na barra lateral
    st.divider()
    # Exibe um cabeçalho para a seção de exportação de dados
    st.header("Export")
    # Cria um seletor para escolher o formato de exportação (CSV ou JSON)
    export_format = st.selectbox("Formato", ["CSV", "JSON"], index=0)
    # Cria um slider para definir a frequência de salvamento de registros (a cada N frames)
    export_every_n = st.slider("Salvar registro a cada N frames", 1, 30, 1, 1)

# Cria uma configuração de predição para o YOLO com os parâmetros selecionados e define o dispositivo como CPU
cfg = PredictConfig(conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, device="cpu")
# Carrega o modelo YOLO usando os pesos selecionados
yolo = load_yolo(yolo_weights)

# Bloco try-except para carregar o classificador Haar cascade para detecção de faces
try:
    # Tenta carregar o classificador Haar cascade
    cascade = get_face_cascade()
# Captura qualquer exceção que ocorra durante o carregamento
except Exception as e:
    # Define cascade como None se houver um erro
    cascade = None
    # Exibe uma mensagem de erro na barra lateral
    st.sidebar.error("Haar cascade não encontrado. Veja o PDF/README para corrigir.")
    # Exibe os detalhes da exceção na barra lateral
    st.sidebar.caption(str(e))

# Cria duas abas na interface principal: "Cadastro" e "Live"
tabA, tabB = st.tabs(["Cadastro (Banco de imagens + nomes)", "Live (Webcam / RTSP + Export)"])

# Conteúdo da aba "Cadastro"
with tabA:
    # Exibe um subtítulo para a aba de cadastro
    st.subheader("Cadastro de pessoas (imagens + nome) — SQLite + Pasta Gallery/")
    # Divide a área da aba em duas colunas, com espaçamento "large"
    col1, col2 = st.columns(2, gap="large")

    with col1:
        # Cria um campo de entrada de texto para o nome da pessoa
        name = st.text_input("Nome da pessoa")
        # Cria um uploader de arquivos para que o usuário possa enviar imagens
        imgs = st.file_uploader(
            "Envie 1 ou mais imagens (jpg/png) com o rosto visível",
            # Define os tipos de arquivo aceitos (JPG, JPEG, PNG)
            type=["jpg", "jpeg", "png"],
            # Permite o upload de múltiplos arquivos
            accept_multiple_files=True
        )

        # Cria um botão para salvar as imagens no banco de dados
        # O botão é desativado se o nome ou as imagens não forem fornecidos
        if st.button("Salvar no banco", disabled=not (name and imgs)):
            # Verifica se o classificador Haar cascade está carregado
            if cascade is None:
                # Exibe uma mensagem de erro se o Haar cascade não for encontrado
                st.error("Sem Haar cascade. Corrija o arquivo assets/haarcascade_frontalface_default.xml e tente de novo.")
            else:
                # Insere ou atualiza o nome da pessoa no banco de dados e obtém o ID
                pid = upsert_person(name.strip())
                # Cria um caminho para o diretório da pessoa dentro da galeria
                person_dir = os.path.join(GALLERY_DIR, f"{pid}_{name.strip().replace(' ', '_')}")
                # Garante que o diretório da pessoa exista
                ensure_dir(person_dir)

                # Inicializa contadores para imagens salvas e ignoradas
                saved = 0
                skipped = 0
                # Itera sobre cada arquivo de imagem enviado
                for f in imgs:
                    # Abre a imagem usando PIL e converte para RGB
                    pil = Image.open(f).convert("RGB")
                    # Converte a imagem PIL para um array NumPy RGB
                    rgb = np.array(pil)
                    # Converte a imagem RGB para BGR (formato do OpenCV)
                    bgr = rgb_to_bgr(rgb)

                    # Converte a imagem BGR para escala de cinza
                    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    # Detecta faces na imagem em escala de cinza usando o Haar cascade
                    faces = cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
                    # Verifica se nenhuma face foi detectada
                    if len(faces) == 0:
                        # Incrementa o contador de imagens ignoradas
                        skipped += 1
                        # Pula para a próxima imagem
                        continue

                    # Gera um timestamp para o nome do arquivo
                    ts = int(time.time() * 1000)
                    # Cria o caminho de saída para a imagem salva
                    out_path = os.path.join(person_dir, f"{ts}.jpg")
                    # Salva a imagem BGR no diretório da pessoa
                    cv2.imwrite(out_path, bgr)
                    # Adiciona o registro da imagem ao banco de dados
                    add_image(pid, out_path)
                    # Incrementa o contador de imagens salvas
                    saved += 1

                # Exibe uma mensagem de sucesso com o número de imagens salvas
                st.success(f"Salvas {saved} imagens para '{name}'.")
                # Se alguma imagem foi ignorada, exibe uma mensagem informativa
                if skipped:
                    st.info(f"{skipped} imagens foram ignoradas (nenhuma face detectada).")

    with col2:
        # Exibe um subtítulo para a seção de pessoas cadastradas
        st.markdown("### Pessoas cadastradas")
        # Obtém a lista de todas as pessoas cadastradas no banco de dados
        people = list_people()
        # Verifica se não há pessoas cadastradas
        if not people:
            # Exibe uma mensagem informativa
            st.info("Nenhuma pessoa cadastrada ainda.")
        else:
            # Itera sobre cada pessoa na lista
            for pid, pname in people:
                # Divide a coluna em duas sub-colunas para nome e botão de exclusão
                c1, c2 = st.columns([3, 1])
                # Exibe o nome e o ID da pessoa na primeira sub-coluna
                c1.write(f"**{pname}** (id={pid})")
                # Cria um botão "Excluir" na segunda sub-coluna
                if c2.button("Excluir", key=f"del_{pid}"):
                    # Chama a função para excluir a pessoa do banco de dados
                    delete_person(pid)
                    # Exibe uma mensagem de aviso
                    st.warning(f"Removido: {pname}")
                    # Reinicia a aplicação Streamlit para atualizar a lista
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
