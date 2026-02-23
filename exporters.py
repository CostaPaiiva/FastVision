import os
import json
from typing import List, Dict, Any

import pandas as pd

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def export_csv(records: List[Dict[str, Any]], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8")

def export_json(records: List[Dict[str, Any]], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
