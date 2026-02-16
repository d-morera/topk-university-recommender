from __future__ import annotations
from pathlib import Path
from src.utils import load_excel_safe
from src.utils import load_qs_excel
import pandas as pd
import numpy as np


# CLEANING
def initial_clean_qs_schema(df: pd.DataFrame) -> list[pd.DataFrame]:
    sheets = df.sheet_names
    sheets.pop(0)

    frames = []

    for sheet in sheets:
        dfR = pd.read_excel(df, sheet_name=sheet, header=10)
        dfR = dfR[["Institution", "Country / Territory","Score"]].copy()
        dfR["Major"] = sheet
        frames.append(dfR)

    return frames


# DATA AUGMENTATION FORMULA
def data_augemntation_formula(last_score: float, gradient: float, i: int, last_valid: int) -> float:
    return last_score * np.exp(-gradient * (i - last_valid))


# DATA AUGMENTATION
# +0.001 -> ultimos valores score mas bajos / -0.001 -> ultimos valores score mas altos (Gradiente)
def augmentation_qs_schema(frames: list[pd.DataFrame], gradient: float = 0.0008) -> pd.DataFrame:

    dfs = []

    for frame in frames:
        last_valid = frame["Score"].last_valid_index()
        last_score = frame.loc[last_valid, "Score"]
        for i in range(last_valid + 1, len(frame)):
            frame.at[i, "Score"] = data_augemntation_formula(last_score, gradient, i, last_valid)
        dfs.append(frame)
    
    df_qs = pd.concat(dfs, ignore_index=True)

    return df_qs


def clean_qs_schema(df: pd.DataFrame) -> pd.DataFrame:
    df["Score"] = df["Score"].round(1)
    df["Institution"] = df["Institution"].str.upper()
    df["Country / Territory"] = df["Country / Territory"].str.upper()
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