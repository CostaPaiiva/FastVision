# Importa o módulo 'os' para interagir com o sistema operacional (caminhos de arquivo, etc.).
import os
# Importa tipos específicos do módulo 'typing' para anotações de tipo, melhorando a legibilidade e a verificação de tipo.
from typing import Dict, List, Tuple, Optional

# Importa a biblioteca OpenCV (cv2) para processamento de imagem e visão computacional.
import cv2
# Importa a biblioteca NumPy (np) para operações com arrays numéricos, essencial para manipulação de imagens.
import numpy as np


# Define o caminho para o arquivo XML do Haar Cascade para detecção de faces.
CASCADE_PATH = os.path.join("assets", "haarcascade_frontalface_default.xml")

# Define uma função para garantir que os arquivos de assets necessários existam.
def ensure_assets() -> None:
    # Cria a pasta 'assets' se ela ainda não existir.
    os.makedirs("assets", exist_ok=True)
    # Verifica se o arquivo do Haar Cascade não existe no caminho especificado.
    if not os.path.exists(CASCADE_PATH):
        # Lança uma exceção FileNotFoundError se o arquivo não for encontrado.
        raise FileNotFoundError(
            # Mensagem de erro indicando o arquivo ausente.
            f"Arquivo não encontrado: {CASCADE_PATH}\n"
            # Dica para o usuário sobre como resolver o problema.
            "Dica: copie o haarcascade_frontalface_default.xml do OpenCV para a pasta assets/."
        )

# Define uma função para obter e carregar o classificador de faces Haar Cascade.
def get_face_cascade() -> cv2.CascadeClassifier:
    # Chama a função para garantir que os assets necessários existam.
    ensure_assets()
    # Inicializa o classificador de cascata com o caminho para o arquivo XML.
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    # Verifica se o classificador de cascata foi carregado corretamente (não está vazio).
    if cascade.empty():
        # Lança uma exceção RuntimeError se o Haar Cascade falhar ao carregar.
        raise RuntimeError("Falha ao carregar Haar Cascade. Verifique o arquivo XML.")
    # Retorna a instância do classificador de cascata carregado.
    return cascade

# Define uma função para detectar faces em uma imagem em tons de cinza usando um classificador Haar Cascade.
def detect_faces_gray(gray: np.ndarray, cascade: cv2.CascadeClassifier) -> List[Tuple[int, int, int, int]]:
    # Detecta faces na imagem em tons de cinza.
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    # Converte o array de faces detectadas para uma lista e a retorna.
    return list(faces)

# Define uma função para pré-processar uma face detectada.
def preprocess_face(gray: np.ndarray, box: Tuple[int, int, int, int], size=(160, 160)) -> np.ndarray:
    # Desempacota as coordenadas e dimensões da caixa delimitadora da face.
    x, y, w, h = box
    # Recorta a região da face da imagem em tons de cinza.
    face = gray[y:y + h, x:x + w]
    # Redimensiona a face recortada para o tamanho especificado (160x160).
    face = cv2.resize(face, size, interpolation=cv2.INTER_AREA)
    # Retorna a imagem da face pré-processada.
    return face

# Define uma função para treinar um reconhecedor facial LBPH (Local Binary Patterns Histograms).
def train_lbph(samples: List[Tuple[np.ndarray, int]]) -> Optional["cv2.face_LBPHFaceRecognizer"]:
    """
    samples: lista de (face_gray_160x160, label_int)
    """
    # Verifica se há amostras suficientes para treinar o reconhecedor.
    if len(samples) < 2:
        # Se não houver amostras suficientes, retorna None.
        return None
    # Cria uma instância do reconhecedor LBPH com parâmetros específicos.
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
    # Extrai as imagens das faces das amostras.
    X = [s[0] for s in samples]
    # Extrai os rótulos das amostras e os converte para um array NumPy de inteiros.
    y = np.array([s[1] for s in samples], dtype=np.int32)
    # Treina o reconhecedor com as imagens das faces e seus respectivos rótulos.
    recognizer.train(X, y)
    # Retorna o reconhecedor treinado.
    return recognizer


# Define uma função para prever faces em um frame BGR.
def predict_faces(
    # O frame de entrada em formato BGR (Blue, Green, Red).
    frame_bgr: np.ndarray,
    # O objeto reconhecedor facial treinado (LBPHFaceRecognizer).
    recognizer,
    # Um dicionário que mapeia rótulos numéricos para nomes de pessoas.
    label_to_name: Dict[int, str],
    # O classificador Haar Cascade para detecção de faces.
    cascade: cv2.CascadeClassifier,
    # O limiar de distância para considerar uma face reconhecida (padrão 70.0).
    threshold: float = 70.0
):
    """
    # A distância do limiar: quanto menor o valor, mais exigente o reconhecimento (valores entre 50 e 90 geralmente funcionam bem).
    threshold: quanto menor, mais exigente (50-90 costuma ser OK).
    # Retorna uma lista de tuplas contendo (nome, distância, (x,y,w,h)).
    Retorna lista de (name, distance, (x,y,w,h))
    """
    # Converte o frame BGR para tons de cinza, pois a detecção e o reconhecimento geralmente usam esta representação.
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Detecta faces na imagem em tons de cinza usando o classificador Haar Cascade.
    faces = detect_faces_gray(gray, cascade)
    # Inicializa uma lista vazia para armazenar os resultados da previsão.
    out = []
    # Itera sobre cada caixa delimitadora de face detectada.
    for box in faces:
        # Pré-processa a região da face recortada (redimensiona para 160x160).
        face = preprocess_face(gray, box)
        # Realiza a previsão usando o reconhecedor treinado, obtendo o rótulo e a distância.
        label, dist = recognizer.predict(face)  # dist menor = melhor
        # Verifica se a distância de confiança está dentro do limiar e se o rótulo é conhecido.
        if dist <= threshold and label in label_to_name:
            # Se a face for reconhecida, adiciona o nome, a distância e a caixa delimitadora à lista de saída.
            out.append((label_to_name[label], float(dist), box))
        # Se a face não for reconhecida (distância alta ou rótulo desconhecido).
        else:
            # Adiciona "desconhecido", a distância e a caixa delimitadora à lista de saída.
            out.append(("desconhecido", float(dist), box))
    # Retorna a lista de resultados da previsão.
    return out
