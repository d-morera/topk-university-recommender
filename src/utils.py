from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

def load_excel_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe: {path}\n\n"
            "Pon los excels en data/raw (local). No se suben a GitHub."
        )
    return pd.read_excel(path)


def load_qs_excel(path: Path) -> pd.DataFrame:
    df_qs = pd.ExcelFile(path)

    return df_qs


