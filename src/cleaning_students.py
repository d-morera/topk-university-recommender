"""
Cleans and standardises the raw students DataFrame schema for modeling and recommendation.
Column names are normalised, key fields are renamed to canonical names, and dtypes
are coerced so all downstream steps can rely on a consistent schema.
"""

from __future__ import annotations
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

# maps normalised raw column names → canonical project names
DEFAULT_MAIN_COLS = {
    "school curriculum": "school_curriculum",
    "school grade":      "school_grade",
    "gpa":               "gpa",
    "toefl (max)":       "toefl",
    "sat (superscore)":  "sat",
    "profile (4-16)":    "profile_score",
    "peak":              "peak",
    "university":        "university",
    "country":           "country",
    "degree":            "degree",
    "admission":         "admitted",
}


def clean_students_schema(
    df_students_raw: pd.DataFrame,
    rename_map: dict[str, str] = DEFAULT_MAIN_COLS,
) -> pd.DataFrame:
    """
    Normalise column names, select relevant columns, and coerce field types.

    Note: profile_score is inverted (20 - raw) so higher = better.
    "NO PLAN" admissions are mapped to NaN; drop them with null_students_schema.
    """
    df = df_students_raw.copy()

    # strip newlines, extra spaces, lowercase
    df.columns = (
        df.columns.astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace("  ", " ", regex=False)
        .str.strip()
        .str.lower()
    )

    # keep only needed columns (drops financial/unnamed noise implicitly)
    df = df[rename_map.keys()]
    df = df.rename(columns=rename_map)

    df['sat_required'] = df['sat'].notna().astype(int)  # 1 if SAT was provided

    # "NO PLAN" → NaN, kept for now and dropped downstream
    df['admitted'] = df['admitted'].map({"SELECTED": 1, "YES": 1, "NO": 0, "NO PLAN": pd.NA}).astype("Int64")

    df['profile_score'] = 20 - df['profile_score']  # invert: higher = stronger profile

    df["peak"] = df["peak"].notna().astype(int)  # any non-null value = has a peak

    df["school_curriculum"] = df["school_curriculum"].astype("string")

    return df


def null_students_schema(df_students_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with missing values in columns required for modeling.

    SAT NaNs are filled with 0 (sat_required already encodes whether it was provided).
    """
    df = df_students_raw.copy()

    df = df.dropna(subset=['degree'])
    df = df.dropna(subset=['admitted'])
    df['sat'] = df['sat'].fillna(0)

    return df