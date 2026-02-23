# FastVision (YOLO + Face Recognition) — Streamlit App

FastVision é um sistema em **Streamlit** para:
- **Detecção de objetos** com **YOLO (Ultralytics)**
- **Reconhecimento facial** com **OpenCV (LBPH + Haar Cascade)**
- **Cadastro de pessoas** e associação de imagens em banco local
- **Listagem** e **exportação** de dados (CSV/JSON)

> Ideal para projetos de visão computacional locais, protótipos rápidos e pipelines de identificação/detecção com interface web.

---

## ✨ Funcionalidades

- ✅ Upload de imagem (e/ou seleção de imagens cadastradas)
- ✅ Detecção de objetos via YOLO (Ultralytics)
- ✅ Detecção/recorte de faces e pré-processamento
- ✅ Treinamento de reconhecimento facial (LBPH)
- ✅ Predição/identificação facial (quando treinado)
- ✅ Cadastro e atualização de pessoas
- ✅ Armazenamento de imagens no banco (e metadados)
- ✅ Exportação de registros para CSV e JSON
- ✅ Interface simples para operar tudo no navegador

---

## 🧱 Stack / Tecnologias

- **Python 3.10+** (recomendado 3.11)
- **Streamlit** (UI)
- **Ultralytics** (YOLO)
- **OpenCV Contrib** (LBPH / `cv2.face`)
- **NumPy / Pandas**
- **Pillow**
- **tqdm**

---

## 📦 Requisitos

Arquivo `requirements.txt` (sugestão final):

> **Atenção:** evite instalar `opencv-python` e `opencv-contrib-python` juntos.
> Se você usa LBPH (`cv2.face`), use **apenas** `opencv-contrib-python`.

```txt
streamlit>=1.30.0
ultralytics>=8.0.0
opencv-contrib-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
tqdm>=4.66.0

🚀 Instalação (recomendado com ambiente virtual)

1) Clonar e entrar na pasta

git clone https://github.com/SEU-USUARIO/FastVision.git
cd FastVision

2) Criar e ativar venv

Windows (PowerShell):

python -m venv .venv
.\.venv\Scripts\activate

Linux/macOS:

python -m venv .venv
source .venv/bin/activate

3) Instalar dependências

python -m pip install -U pip
python -m pip install -r requirements.txt

Se você já instalou OpenCV duplicado, limpe e reinstale:

python -m pip uninstall -y opencv-python opencv-contrib-python
python -m pip install opencv-contrib-python

▶️ Como rodar (IMPORTANTE)

Use sempre este comando (ele garante que o Streamlit rode no MESMO Python do seu ambiente):

python -m streamlit run app.py

Acesse:

http://localhost:8501

✅ Isso resolve o erro clássico:

ModuleNotFoundError: No module named 'ultralytics'
quando você instala pacotes num Python/venv, mas roda o streamlit de outro.

🗂️ Estrutura do projeto (esperada)
FastVision/

├─ app.py                 # UI Streamlit

├─ db.py                  # Banco local (init, upsert, listagens, imagens)

├─ yolo_backend.py        # YOLODetector + PredictConfig

├─ face_recog.py          # Haar cascade, preprocess, treino LBPH, predição

├─ exporters.py           # Exportação CSV/JSON

├─ requirements.txt

├─ data/                  # (opcional) imagens/modelos/cache

└─ README.md

🧠 Como funciona (visão geral do fluxo)
1) Inicialização

Ao abrir o app, o sistema chama init_db() para preparar o banco local e tabelas necessárias.

2) Cadastro de pessoas

O usuário cadastra uma pessoa (nome / identificador), permitindo:

organizar dataset

treinar reconhecimento facial

associar imagens posteriormente

3) Processamento de imagem

Ao enviar uma imagem:

YOLO detecta objetos (classes, bounding boxes, confiança)

Face pipeline detecta/recorta faces e prepara para treino/predição

4) Treinamento LBPH

Com imagens associadas a pessoas, o sistema:

extrai faces

treina um modelo LBPH para reconhecimento

5) Predição

Com modelo treinado:

reconhece a face mais provável

retorna id/nome e score (dependendo da implementação)

6) Persistência

O sistema pode salvar:

pessoa

imagem

metadados (ex: resultados YOLO, bounding boxes etc.)

7) Exportação

Exporta registros para:

CSV (rápido para Excel/Sheets)

JSON (integração e automações)

⚙️ Configurações (YOLO / PredictConfig)

O yolo_backend.py expõe:

YOLODetector → inicializa modelo e executa predição

PredictConfig → configura parâmetros da predição

Parâmetros típicos (podem variar conforme seu código):

conf (threshold de confiança)

iou (NMS IoU)

classes (filtrar classes)

max_det (máximo de detecções)

imgsz (tamanho da imagem)

Se você colar o conteúdo do PredictConfig, eu documento os campos exatos aqui com exemplos.

🗃️ Banco de dados

O módulo db.py gerencia:

init_db → cria/valida tabelas

upsert_person → cria/atualiza pessoas

add_image → adiciona imagem vinculada

list_people, list_images → consultas para UI

Onde fica o banco?

Depende do seu db.py. Normalmente fica:

no mesmo diretório do projeto, ex: fastvision.db

ou em data/fastvision.db

Se você colar o db.py, eu escrevo aqui o caminho real e o schema das tabelas.

📤 Exportação

O módulo exporters.py geralmente oferece:

export_csv(...)

export_json(...)

Sugestão: exportar por filtros

por pessoa

por data

por tipo (faces / objetos)

🧯 Troubleshooting (erros comuns)
1) No module named 'ultralytics'

Você instalou num ambiente e rodou o Streamlit em outro.

✅ Solução:

python -m pip install ultralytics
python -m streamlit run app.py
2) cv2.face não existe

Você está sem OpenCV Contrib.

✅ Solução:

python -m pip uninstall -y opencv-python
python -m pip install opencv-contrib-python
3) Conflito OpenCV (opencv-python + opencv-contrib-python)

✅ Mantenha só opencv-contrib-python.

4) Erros relacionados a torch/YOLO (CPU/GPU)

O Ultralytics depende de torch. Em alguns ambientes (principalmente Windows) pode precisar ajuste.
Se aparecer traceback com torch, cole o erro completo aqui que eu te passo o comando correto (CPU ou CUDA).

🧪 Dicas de uso/qualidade

Use imagens bem iluminadas para reconhecimento facial

Para LBPH:

mais amostras por pessoa = melhor

normalize tamanho/cinza no preprocess_face

Para YOLO:

ajuste conf e iou para reduzir falsos positivos

use classes se quiser filtrar apenas algumas classes

✅ Recomendações de “produção”

Criar .streamlit/config.toml para UI:

[server]
headless = true
port = 8501
enableCORS = false

[browser]
gatherUsageStats = false

Adicionar .gitignore:

.venv/

__pycache__/

*.pyc

*.db

data/

outputs/

.streamlit/secrets.toml

🗺️ Roadmap (ideias)

 Suporte a webcam/stream (tempo real)

 Batch upload e processamento em lote

 Dashboard com estatísticas (classes detectadas, pessoas reconhecidas)

 Exportação com filtros e relatórios

 Cache de modelo YOLO e resultados (melhora performance)