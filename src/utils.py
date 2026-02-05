from __future__ import annotations
from pathlib import Path
import pandas as pd

def load_excel_safe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"No existe: {path}\n\n"
            "Pon los excels en data/raw (local). No se suben a GitHub."
        )
    return pd.read_excel(path)

def quick_profile(df: pd.DataFrame, name: str, n: int = 5) -> None:
    print(f"=== {name} ===")
    print("shape:", df.shape)
    print("columns:", list(df.columns))
    print("\nnulls (top 15):")
    display(df.isna().mean().sort_values(ascending=False).head(15))
    print("\nhead:")
    display(df.head(n))