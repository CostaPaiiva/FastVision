# Importa o decorador dataclass para criar classes de dados
from dataclasses import dataclass
# Importa tipos para anotação de tipos, incluindo List, Optional, Dict, Any e Tuple
from typing import List, Optional, Dict, Any, Tuple

# Importa a biblioteca numpy para manipulação de arrays
import numpy as np
# Importa a classe YOLO da biblioteca ultralytics
from ultralytics import YOLO

# Define uma classe de dados para configurar os parâmetros de previsão
@dataclass
class PredictConfig:
    # Confiança mínima para detecção (padrão: 0.25)
    conf: float = 0.25
    # Limiar de Intersection Over Union (IOU) para Non-Maximum Suppression (NMS) (padrão: 0.45)
    iou: float = 0.45
    # Tamanho da imagem para inferência (padrão: 640 pixels)
    imgsz: int = 640
    # Número máximo de detecções por imagem (padrão: 300)
    max_det: int = 300
    # Lista opcional de IDs de classes para filtrar as detecções (padrão: None, todas as classes)
    classes: Optional[List[int]] = None
    # Dispositivo para executar a inferência ('cpu' ou 'cuda:0', etc.) (padrão: "cpu")
    device: str = "cpu"
    # Se deve usar precisão de ponto flutuante de metade (FP16) (padrão: False)
    half: bool = False
    # Se deve usar NMS agnóstico de classe (ignora a classe na NMS) (padrão: False)
    agnostic_nms: bool = False

# Define a classe YOLODetector para encapsular o modelo YOLO
class YOLODetector:
    # O construtor da classe, que carrega o modelo YOLO
    def __init__(self, weights_path: str):
        # Inicializa o modelo YOLO com o caminho dos pesos fornecido
        self.model = YOLO(weights_path)

    # Método para realizar a previsão em um frame de imagem
    def predict(self, frame_bgr: np.ndarray, cfg: PredictConfig) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Retorna:
          - frame anotado (BGR)
          - lista de detecções (uma por box)
        """
        # Executa a previsão do modelo YOLO no frame_bgr usando as configurações fornecidas
        results = self.model.predict(
            # A imagem de entrada para a previsão
            source=frame_bgr,
            # Limiar de confiança para detecções
            conf=cfg.conf,
            # Limiar de IOU para NMS
            iou=cfg.iou,
            # Tamanho da imagem para inferência
            imgsz=cfg.imgsz,
            # Número máximo de detecções
            max_det=cfg.max_det,
            # Classes a serem consideradas
            classes=cfg.classes,
            # Dispositivo para inferência
            device=cfg.device,
            # Usar precisão de metade
            half=cfg.half,
            # Usar NMS agnóstico
            agnostic_nms=cfg.agnostic_nms,
            # Não imprimir saída verbosa
            verbose=False
        )
        # Pega o primeiro resultado da lista de resultados (geralmente há apenas um para uma única imagem)
        r = results[0]
        # Desenha as detecções no frame e retorna o frame anotado
        annotated = r.plot()

        # Inicializa uma lista vazia para armazenar as detecções formatadas
        dets: List[Dict[str, Any]] = []
        # Verifica se há caixas de detecção no resultado
        if r.boxes is not None and len(r.boxes) > 0:
            # Extrai as coordenadas das caixas (x1, y1, x2, y2) e converte para numpy array na CPU
            xyxy = r.boxes.xyxy.cpu().numpy()
            # Extrai as confianças das detecções e converte para numpy array na CPU
            confs = r.boxes.conf.cpu().numpy()
            # Extrai os IDs das classes, converte para numpy array na CPU e para tipo inteiro
            clss = r.boxes.cls.cpu().numpy().astype(int)
            # Obtém os nomes das classes do modelo
            names = r.names
            # Itera sobre as coordenadas, confianças e IDs de classe
            for (x1, y1, x2, y2), c, k in zip(xyxy, confs, clss):
                # Adiciona um dicionário com os detalhes da detecção à lista 'dets'
                dets.append({
                    # ID da classe da detecção
                    "class_id": int(k),
                    # Nome da classe da detecção
                    "class_name": str(names[int(k)]),
                    # Pontuação de confiança da detecção
                    "confidence": float(c),
                    # Coordenadas da caixa delimitadora (x1, y1, x2, y2)
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                })

        # Retorna o frame anotado e a lista de detecções
        return annotated, dets
