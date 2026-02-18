from __future__ import annotations

import os
import sys
from pathlib import Path

"""
Baixa automaticamente o yolov8n.pt usando a própria biblioteca ultralytics.

Observações:
- Este script usa internet APENAS no momento do download (setup).
- O runtime do backend não depende de web.
- Se você preferir, baixe manualmente e coloque em: backend/weights/yolov8n.pt
"""

def main() -> int:
    weights_dir = Path(__file__).resolve().parents[1] / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    target = weights_dir / "yolov8n.pt"

    if target.exists() and target.stat().st_size > 0:
        print(f"[ok] Já existe: {target}")
        return 0

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("[erro] ultralytics não está instalado. Rode: pip install -r requirements.txt")
        print(e)
        return 1

    print("[info] Baixando yolov8n.pt via Ultralytics...")
    try:
        _ = YOLO("yolov8n.pt")  # baixa para cache do ultralytics
    except Exception as e:
        print("[erro] Falha ao baixar via ultralytics.")
        print(e)
        return 1

    # O arquivo normalmente fica no cache do ultralytics/torch.
    # Tentamos localizar e copiar para backend/weights/yolov8n.pt
    candidates = []

    # Possíveis locais comuns
    home = Path.home()
    candidates += list(home.glob("**/yolov8n.pt"))

    # Filtra por tamanho plausível (> 1MB)
    candidates = [p for p in candidates if p.is_file() and p.stat().st_size > 1_000_000]

    if not candidates:
        print("[warn] Não consegui localizar o arquivo no disco automaticamente.")
        print("Baixe manualmente e coloque em backend/weights/yolov8n.pt")
        return 1

    # Pega o maior candidato
    src = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)[0]
    target.write_bytes(src.read_bytes())
    print(f"[ok] Copiado para: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
