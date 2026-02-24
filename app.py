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
APP_TITLE = "Fast Vision (YOLO + Reconhecimento por Nome)"
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
    # Exibe um subtítulo para a aba "Live"
    st.subheader("Live: Webcam ou RTSP (OpenCV) + export por frame")
    # Exibe uma dica para otimização de desempenho em CPU
    st.caption("Dica CPU: use yolo11n + imgsz 416/480 para ficar leve.")

    # Cria um seletor de rádio para escolher a fonte de vídeo (Webcam ou RTSP)
    source_type = st.radio("Fonte", ["Webcam", "RTSP"], horizontal=True)
    # Cria um campo numérico para o índice da webcam, desabilitado se a fonte não for "Webcam"
    cam_index = st.number_input("Indice da webcam", 0, 10, 0, 1, disabled=(source_type != "Webcam"))
    # Cria um campo de texto para a URL RTSP, desabilitado se a fonte não for "RTSP"
    rtsp_url = st.text_input("RTSP URL (ex: rtsp://user:pass@ip:554/...) ", disabled=(source_type != "RTSP"))

    # Divide a área em duas colunas com proporção 2:1
    colL, colR = st.columns([2, 1], gap="large")

    with colR:
        # Exibe um subtítulo para a seção de controles
        st.markdown("### Controles")
        # Cria um toggle (chave) para iniciar/parar a execução
        start = st.toggle("Iniciar", value=False)
        # Cria um checkbox para decidir se o vídeo anotado será gravado
        save_annotated_video = st.checkbox("Gravar video anotado (mp4)", value=False)
        # Cria um slider para definir o tempo máximo de execução em segundos
        max_seconds = st.slider("Limite (segundos) para rodar", 5, 300, 30, 5)
        # Cria um campo de texto para o nome da execução, preenchido com um timestamp padrão
        run_name = st.text_input("Nome da execucao", value=time.strftime("%Y%m%d-%H%M%S"))

        # Exibe um subtítulo para a seção de reconhecimento
        st.markdown("### Reconhecimento")
        # Cria um checkbox para ativar/desativar o reconhecimento facial por nome
        use_face = st.checkbox("Ativar reconhecimento por nome", value=True)
        # Verifica se o reconhecimento facial está ativado mas o modelo não foi treinado
        if use_face and "recognizer" not in st.session_state:
            # Exibe um aviso para treinar o modelo primeiro
            st.warning("Treine o modelo na aba Cadastro antes de usar reconhecimento.")

    with colL:
        # Cria um placeholder vazio na interface do Streamlit para exibir o frame do vídeo
        frame_box = st.empty()
        # Cria outro placeholder vazio para exibir informações sobre o processamento
        info_box = st.empty()

    # Verifica se o botão "Iniciar" foi ativado
    if start:
        # Verifica se o classificador Haar cascade para detecção de faces foi carregado com sucesso
        if cascade is None:
            # Se não foi, exibe uma mensagem de erro
            st.error("Sem Haar cascade. Corrija o arquivo assets/haarcascade_frontalface_default.xml.")
        else:
            # Determina a fonte do vídeo (índice da webcam ou URL RTSP) com base na seleção do usuário
            src = int(cam_index) if source_type == "Webcam" else rtsp_url.strip()
            # Tenta abrir a fonte de vídeo usando OpenCV (VideoCapture)
            cap = cv2.VideoCapture(src)
            # Verifica se a fonte de vídeo foi aberta com sucesso
            if not cap.isOpened():
                # Se não conseguiu abrir, exibe uma mensagem de erro
                st.error("Nao consegui abrir a fonte. Verifique webcam/RTSP e permissoes.")
            else:
                # Inicializa uma lista vazia para armazenar os registros de detecção e reconhecimento
                records: List[Dict[str, Any]] = []
                # Constrói o caminho para o diretório de saída desta execução específica
                out_dir = os.path.join(RUNS_DIR, run_name)
                # Garante que o diretório de saída exista, criando-o se necessário
                ensure_dir(out_dir)

                # Inicializa o objeto gravador de vídeo como None
                writer = None
                # Inicializa o caminho do vídeo de saída como None
                out_video = None
                # Verifica se a opção de salvar o vídeo anotado está ativada
                if save_annotated_video:
                    # Obtém a largura do frame da fonte de vídeo, com fallback para 640
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                    # Obtém a altura do frame da fonte de vídeo, com fallback para 480
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
                    # Obtém a taxa de quadros por segundo (FPS) da fonte de vídeo, com fallback para 20.0
                    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
                    # Define o caminho completo para o arquivo de vídeo de saída
                    out_video = os.path.join(out_dir, "annotated.mp4")
                    # Define o codec de vídeo (FourCC) para MP4
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    # Cria um objeto VideoWriter para gravar o vídeo anotado
                    writer = cv2.VideoWriter(out_video, fourcc, float(fps), (w, h))

                # Registra o tempo inicial da execução para controle do limite de segundos
                t0 = time.time()
                # Inicializa o contador de frames processados
                frame_idx = 0

                # Inicia um loop infinito para processar os frames da fonte de vídeo
                while True:
                    # Lê um frame da fonte de vídeo (cap é o objeto VideoCapture)
                    ok, frame = cap.read()
                    # Verifica se a leitura do frame foi bem-sucedida
                    if not ok:
                        # Se não foi, exibe um aviso e encerra o loop
                        info_box.warning("Falha ao ler frame. Encerrando.")
                        break

                    # Realiza a predição de objetos YOLO no frame e obtém o frame anotado e as detecções
                    annotated, dets = yolo.predict(frame, cfg)

                    # Inicializa uma lista vazia para armazenar os resultados do reconhecimento facial
                    faces_out = []
                    # Verifica se o reconhecimento facial está ativado e se o reconhecedor está disponível na sessão
                    if use_face and "recognizer" in st.session_state:
                        # Chama a função predict_faces para detectar e reconhecer faces no frame
                        faces_out = predict_faces(
                            frame_bgr=frame, # O frame atual em formato BGR
                            recognizer=st.session_state["recognizer"], # O modelo de reconhecimento treinado
                            label_to_name=st.session_state.get("label_to_name", {}), # Mapeamento de IDs para nomes
                            cascade=cascade, # O classificador Haar cascade para detecção de faces
                            threshold=face_threshold # Limiar de confiança para o reconhecimento
                        )
                        # Itera sobre cada resultado de reconhecimento facial obtido
                        for fname, dist, (x, y, w, h) in faces_out:
                            # Desenha um retângulo verde ao redor da face reconhecida no frame anotado
                            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            # Adiciona o nome da pessoa e a distância (confiança) acima do retângulo da face
                            cv2.putText(
                                annotated, # Imagem onde desenhar
                                f"{fname} ({dist:.0f})", # Texto a ser exibido (nome e distância)
                                (x, max(0, y - 8)), # Posição do texto (ajusta para não sair da imagem)
                                cv2.FONT_HERSHEY_SIMPLEX, # Fonte do texto
                                0.6, # Escala da fonte
                                (0, 255, 0), # Cor do texto (verde)
                                2 # Espessura da linha do texto
                            )

                    # Verifica se o índice do frame atual é um múltiplo do valor de export_every_n
                    if frame_idx % export_every_n == 0:
                        # Obtém o timestamp atual
                        ts = time.time()
                        # Itera sobre cada detecção YOLO
                        for d in dets:
                            # Adiciona um novo registro à lista 'records' com informações da detecção YOLO
                            records.append({
                                "run": run_name, # Nome da execução
                                "frame_idx": frame_idx, # Índice do frame
                                "timestamp": ts, # Timestamp
                                "type": "yolo", # Tipo de detecção (YOLO)
                                **d # Desempacota os detalhes da detecção YOLO
                            })
                        # Itera sobre cada resultado de reconhecimento facial
                        for fname, dist, (x, y, w, h) in faces_out:
                            # Adiciona um novo registro à lista 'records' com informações do reconhecimento facial
                            records.append({
                                "run": run_name, # Nome da execução
                                "frame_idx": frame_idx, # Índice do frame
                                "timestamp": ts, # Timestamp
                                "type": "face", # Tipo de detecção (face)
                                "name": fname, # Nome da pessoa reconhecida
                                "distance": float(dist), # Distância (confiança) do reconhecimento
                                "x1": int(x), "y1": int(y), "x2": int(x + w), "y2": int(y + h) # Coordenadas da face
                            })

                    # Se um gravador de vídeo estiver ativo
                    if writer is not None:
                        # Escreve o frame anotado no arquivo de vídeo
                        writer.write(annotated)

                    # Exibe o frame anotado na interface do Streamlit (convertendo de BGR para RGB)
                    frame_box.image(bgr_to_rgb(annotated), use_container_width=True)
                    # Exibe informações sobre o frame atual (índice, número de detecções YOLO e faces)
                    info_box.info(f"Frame: {frame_idx} | YOLO: {len(dets)} | Faces: {len(faces_out)}")

                    # Incrementa o índice do frame
                    frame_idx += 1
                    # Verifica se o tempo de execução excedeu o limite máximo definido
                    if time.time() - t0 >= max_seconds:
                        # Sai do loop se o tempo limite for atingido
                        break

                # Libera o objeto de captura de vídeo (webcam/RTSP)
                cap.release()
                # Se um gravador de vídeo estava ativo
                if writer is not None:
                    # Libera o gravador de vídeo
                    writer.release()

                # Verifica o formato de exportação selecionado
                if export_format == "CSV":
                    # Define o caminho de saída para o arquivo CSV
                    out_path = os.path.join(out_dir, "detections.csv")
                    # Exporta os registros para um arquivo CSV
                    export_csv(records, out_path)
                else:
                    # Define o caminho de saída para o arquivo JSON
                    out_path = os.path.join(out_dir, "detections.json")
                    # Exporta os registros para um arquivo JSON
                    export_json(records, out_path)

                # Exibe uma mensagem de sucesso indicando onde a execução foi salva
                st.success(f"Execucao salva em: {out_dir}")
                # Exibe o caminho do arquivo de exportação
                st.write("Arquivo export:", out_path)
                # Se um vídeo de saída foi gravado e existe
                if out_video and os.path.exists(out_video):
                    # Exibe o vídeo de saída no Streamlit
                    st.video(out_video)
