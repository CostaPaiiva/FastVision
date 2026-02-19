# CV App Local (YOLO + Reconhecimento por Nome) — Streamlit (Windows/CPU)

Aplicação **100% local** (sem API externa) em **Python + Streamlit** que combina **detecção de objetos com YOLO (Ultralytics)** e **reconhecimento facial por identidade** via **OpenCV (LBPH)** com **SQLite** para cadastro de pessoas (nome + múltiplas imagens), suportando **Webcam (Windows)** e **RTSP (câmera IP)**, além de **export por frame em CSV/JSON**.

---

## Features

- **Detecção de objetos (YOLO)** com ajuste de `conf`, `iou`, `imgsz`, `max_det`
- **Reconhecimento por nome (faces)**:
  - Cadastro de pessoas (nome + várias imagens)
  - Treino local do reconhecedor **LBPH**
  - Identificação em tempo real na live (Webcam/RTSP)
- **Fontes de vídeo**:
  - Webcam (Windows, via OpenCV)
  - RTSP (câmera IP / NVR/DVR)
- **Export por frame**:
  - `CSV` ou `JSON` com registros de detecções por frame (objetos e faces)
- **Totalmente offline**:
  - SQLite local (`data/app.db`) + arquivos em `gallery/` e `runs/`

---

## Stack

- **Streamlit** (UI web local)
- **Ultralytics YOLO** (inferência)
- **OpenCV + LBPHFaceRecognizer** (reconhecimento facial)
- **SQLite** (cadastro de pessoas + paths de imagens)
- **Pandas** (export CSV)

---

## Requisitos

- **Windows 10/11**
- **Python 3.10+** (recomendado)
- CPU (sem GPU)

---

## Instalação (Windows)

No PowerShell, dentro da pasta do projeto:

```bat
python -m venv .venv
.venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt
Haar Cascade (Obrigatório para o reconhecimento facial)
Você precisa do arquivo:

assets/haarcascade_frontalface_default.xml

Como obter
Geralmente ele já vem instalado com o OpenCV. Procure algo como:

...\.venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml

Copie para:

cv_app_local/assets/haarcascade_frontalface_default.xml

Se o arquivo não estiver, baixe do repositório oficial do OpenCV e coloque em assets/.

Como Rodar
.venv\Scripts\activate
streamlit run app.py
Acesse:

http://localhost:8501

Uso
1) Cadastro de Pessoas (Banco de imagens + nome)
Vá na aba Cadastro

Digite o nome da pessoa

Faça upload de várias imagens com o rosto visível (boa iluminação, face frontal)

Clique em Salvar no banco

Clique em Treinar/Atualizar modelo de reconhecimento

Dicas para melhorar o reconhecimento (LBPH):

Use 10+ fotos por pessoa

Varie levemente ângulo e iluminação

Rostos grandes e nítidos funcionam melhor

2) Live (Webcam / RTSP)
Vá na aba Live

Selecione:

Webcam (índice 0, 1, 2…) ou

RTSP (cole a URL)

Ajuste:

YOLO: conf, iou, imgsz

Faces: threshold (menor = mais exigente)

Clique em Iniciar

Ao final, o app salva:

runs/<run_name>/detections.csv ou .json

(Opcional) vídeo anotado runs/<run_name>/annotated.mp4

RTSP (Exemplos)
O path muda conforme a marca (exemplos comuns):

rtsp://usuario:senha@IP:554/stream1

rtsp://usuario:senha@IP:554/h264

rtsp://usuario:senha@IP:554/cam/realmonitor?channel=1&subtype=0

Se não conectar: teste subtype=1 (substream) para reduzir carga na CPU.

Export (CSV/JSON por frame)
Os exports são salvos em:

runs/<execucao>/detections.csv ou detections.json

Formato (lógica)
Cada detecção vira uma linha/registro com:

run, frame_idx, timestamp, type

Para YOLO: class_id, class_name, confidence, x1,y1,x2,y2

Para faces: name, distance, x1,y1,x2,y2

Performance (CPU)
Para rodar melhor no CPU:

Use yolo11n.pt

imgsz = 416 ou 480

Aumente export_every_n (ex.: salvar a cada 2–5 frames)

Troubleshooting
Webcam não abre
Tente trocar o índice (0/1/2)

Feche apps que possam estar usando a câmera (Teams/Zoom)

Verifique permissões de câmera no Windows

RTSP não abre
Confirme IP/porta/usuário/senha

Teste o stream com VLC antes

Tente substream (menor resolução)

Reconhecimento sempre “desconhecido”
Cadastre mais fotos por pessoa

Melhore iluminação

Ajuste threshold (ex.: 70 → 85 para ficar menos exigente)

Erro com cv2.face
Garanta que instalou opencv-contrib-python

Reinstale:

pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-contrib-python
Segurança e Privacidade
Tudo roda localmente.

O “banco” é um SQLite local (data/app.db) e imagens ficam em gallery/.

Use apenas imagens e fontes de vídeo com autorização.

Roadmap (ideias)
Tracking (ByteTrack) e contagem por linha/área

Export em Parquet e dashboards

Treino com embeddings (FaceNet/ArcFace) para maior robustez

Modo “dataset builder” automático via webcam

Créditos
Ultralytics YOLO (inferência)

OpenCV (detecção/visão e reconhecimento LBPH)

Streamlit (UI)

