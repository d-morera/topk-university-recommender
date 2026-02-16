"""
    Standardize students dataframe schema:
      - normalize column names (lowercase, remove newlines/double spaces)
      - drop irrelevant columns (ignore if missing)
      - rename key columns to stable names (ignore if missing)

    Returns a cleaned copy (does not mutate input).
"""

from __future__ import annotations
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

# Column rename map (raw -> clean/professional)
# Column rename map AFTER normalization to lowercase / stripped / no '\n'
DEFAULT_MAIN_COLS = {
    "school curriculum": "school_curriculum",
    "school grade": "school_grade",
    "gpa": "gpa",
    "toefl (max)": "toefl",
    "sat (superscore)": "sat",
    "profile (4-16)": "profile_score",
    "peak": "peak",
    "university": "university",
    "country": "country",
    "degree": "degree",
    "admission": "admitted",
}

def clean_students_schema(
    df_students_raw: pd.DataFrame,
    rename_map: dict[str, str] = DEFAULT_MAIN_COLS,
) -> pd.DataFrame:

    df = df_students_raw.copy()

    # normalizar columnas
    cols = (
        df.columns.astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace("  ", " ", regex=False)
        .str.strip()
        .str.lower()
    )

    df.columns = cols

    df = df[rename_map.keys()]

    # renombrar columnas
    df = df.rename(columns=rename_map)

    # nueva columna: si se requiere SAT
    df['sat_required'] = df['sat'].notna().astype(int)

    # valorizar
    df['admitted'] = df['admitted'].map({"NO": 0, 'SELECTED': 1, "YES": 1})
    df['profile_score'] = 20 - df['profile_score']

    return df


def null_students_schema(df_students_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_students_raw.copy()

    # borrar nulos en columnas clave
    df = df.dropna(subset=['degree'])
    df = df.dropna(subset=['admitted'])
    df['sat'] = df['sat'].fillna(0)

    for i in range(len(df["peak"].unique())):
            df["peak"] = df["peak"].replace(df["peak"].unique()[i], i)

    return df