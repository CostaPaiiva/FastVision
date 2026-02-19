# CV App Local (YOLO + Reconhecimento por Nome)

App web local (Streamlit) para:
- Detecção de objetos com YOLO (Ultralytics)
- Reconhecimento por nome (faces) com OpenCV LBPH (CPU)
- Webcam e RTSP
- Export CSV/JSON por frame
- Banco local SQLite (arquivo `data/app.db`)

## Requisitos
- Windows 10/11
- Python 3.10+ recomendado

## Como rodar
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
streamlit run app.py
```

Abra: http://localhost:8501

## Observacoes
- No CPU, use `yolo11n.pt` e `imgsz` 416/480 para melhor performance.
- O arquivo `assets/haarcascade_frontalface_default.xml` ja vem no zip.
