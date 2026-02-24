# Importa o módulo 'os' para interagir com o sistema operacional, como criar diretórios.
import os
# Importa o módulo 'json' para trabalhar com dados JSON.
import json
# Importa tipos específicos do módulo 'typing' para anotações de tipo.
from typing import List, Dict, Any

# Importa a biblioteca pandas, comumente usada para análise e manipulação de dados, especialmente com DataFrames.
import pandas as pd

# Define uma função para garantir que um diretório exista.
def ensure_dir(path: str) -> None:
    # Cria recursivamente os diretórios no caminho especificado se eles não existirem.
    os.makedirs(path, exist_ok=True)

# Define uma função para exportar uma lista de dicionários para um arquivo CSV.
def export_csv(records: List[Dict[str, Any]], path: str) -> None:
    # Garante que o diretório de destino para o arquivo CSV exista.
    ensure_dir(os.path.dirname(path))
    # Converte a lista de dicionários em um DataFrame do pandas e o exporta para CSV.
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8")

# Define uma função para exportar uma lista de dicionários para um arquivo JSON.
def export_json(records: List[Dict[str, Any]], path: str) -> None:
    # Garante que o diretório de destino para o arquivo JSON exista.
    ensure_dir(os.path.dirname(path))
    # Abre o arquivo no modo de escrita com codificação UTF-8.
    with open(path, "w", encoding="utf-8") as f:
        # Serializa a lista de dicionários para JSON e a escreve no arquivo, formatando com indentação.
        json.dump(records, f, ensure_ascii=False, indent=2)
