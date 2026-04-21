"""
Extracts and standardises QS World University Rankings from a multi-sheet
Excel workbook into a single flat DataFrame ready for merging and recommendation.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils import load_excel_safe
from src.utils import load_qs_excel


def initial_clean_qs_schema(df) -> list[pd.DataFrame]:
    """
    Read each subject sheet from the QS ExcelFile and return a list of
    per-subject DataFrames with columns [Institution, Country / Territory, Score, Major].

    Note: expects a pd.ExcelFile object, not a plain DataFrame.
    The first sheet (cover/intro) is dropped. header=10 because the table
    starts after 10 metadata rows.
    """
    sheets = df.sheet_names
    sheets.pop(0)  # first sheet is a cover page, not a ranking

    frames = []

    for sheet in sheets:
        dfR = pd.read_excel(df, sheet_name=sheet, header=10)  # table starts at row 11
        dfR = dfR[["Institution", "Country / Territory", "Score"]].copy()
        dfR["Major"] = sheet
        frames.append(dfR)

    return frames


def data_augemntation_formula(last_score: float, gradient: float, i: int, last_valid: int) -> float:
    """Exponential decay from the last known QS score — heuristic, not official QS data."""
    return last_score * np.exp(-gradient * (i - last_valid))


def augmentation_qs_schema(frames: list[pd.DataFrame], gradient: float = 0.0008) -> pd.DataFrame:
    """
    Fill missing tail scores with exponential decay and concatenate all sheets.

    QS sheets often have NaNs below a certain rank. gradient=0.0008 gives a
    slow, gentle drop-off — increase it to penalise lower ranks more sharply.
    """
    dfs = []

    for frame in frames:
        last_valid = frame["Score"].last_valid_index()
        last_score = frame.loc[last_valid, "Score"]

        # fill every NaN below the last official score
        for i in range(last_valid + 1, len(frame)):
            frame.at[i, "Score"] = data_augemntation_formula(last_score, gradient, i, last_valid)
        dfs.append(frame)

    df_qs = pd.concat(dfs, ignore_index=True)
    return df_qs


def clean_qs_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise text fields and rename to canonical schema
    (university / country / qs_score / degree).

    Country harmonisation covers the four variants most common in the QS 2025
    workbook (USA full name, China Mainland, Venezuela, Iran).
    """
    df["Score"] = df["Score"].round(1)
    df["Institution"] = df["Institution"].str.upper()
    df["Country / Territory"] = df["Country / Territory"].str.upper()

    # normalise country name variants to match student dataset
    df.loc[df['Country / Territory'] == 'UNITED STATES OF AMERICA', 'Country / Territory'] = 'UNITED STATES'
    df.loc[df['Country / Territory'] == 'CHINA (MAINLAND)', 'Country / Territory'] = 'CHINA'
    df.loc[df['Country / Territory'] == 'VENEZUELA (BOLIVARIAN REPUBLIC OF)', 'Country / Territory'] = 'VENEZUELA'
    df.loc[df['Country / Territory'] == 'IRAN (ISLAMIC REPUBLIC OF)', 'Country / Territory'] = 'IRAN'

    df.rename(columns={
        "Institution": "university",
        "Country / Territory": "country",
        "Score": "qs_score",
        "Major": "degree"
    }, inplace=True)

    return df