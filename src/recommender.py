from __future__ import annotations
import pandas as pd

def build_candidate_catalog(
    df_qs: pd.DataFrame,
    degree: str | None = None,
    country: str | None = None,
    uni_col: str = "university",
    degree_col: str = "degree",
    country_col: str = "country",
    qs_col: str = "qs_score",
) -> pd.DataFrame:
    """
    Filters QS catalog by degree and/or country and returns the columns needed
    for recommendation scoring.
    """
    df = df_qs.copy()

    if degree is not None:
        df = df[df[degree_col] == degree]

    if country is not None:
        df = df[df[country_col] == country]

    cols = [uni_col, country_col, degree_col, qs_col]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].drop_duplicates()

    return df



# Esto es para que normalizar de 0-1, y se hace por degree ya que en cada degree 
# hay una distribucion diferente de qs_score
def add_qs_normalized(
    df_candidates: pd.DataFrame,
    qs_col: str = "qs_score",
    degree_col: str = "degree",
) -> pd.DataFrame:
    """
    Adds qs_norm in [0,1] computed within each degree (mapped_major).
    """
    df = df_candidates.copy()

    def _minmax(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mn == mx:
            return pd.Series([0.5] * len(s), index=s.index)  # neutral
        return (s - mn) / (mx - mn)

    df["qs_norm"] = df.groupby(degree_col)[qs_col].transform(_minmax)
    return df


def score_candidates_with_model(
    model,
    student_row: pd.Series | dict,
    df_candidates: pd.DataFrame,
    student_feature_cols: list[str],
) -> pd.DataFrame:
    """
    Replicates student features across all candidates, predicts p_admit using model,
    and returns candidates with p_admit.
    """
    if isinstance(student_row, dict):
        student_row = pd.Series(student_row)

    # base: candidate columns
    df = df_candidates.copy()

    # replicate student features
    for c in student_feature_cols:
        df[c] = student_row.get(c, None)

    # Predict probability of admission
    df["p_admit"] = model.predict_proba(df[student_feature_cols + [c for c in df_candidates.columns if c not in student_feature_cols]])[:, 1]
    return df


def recommend_top_k(
    model,
    student_row: pd.Series | dict,
    df_qs: pd.DataFrame,
    feature_cols: list[str],
    degree: str | None = None,
    country: str | None = None,
    alpha: float = 0.7,
    k: int = 10,
    threshold: float | None = None,
    uni_col: str = "university",
    degree_col: str = "degree",
    country_col: str = "country",
    qs_col: str = "qs_score",
) -> pd.DataFrame:
    """
    Build QS candidate set (filtered), compute p_admit with model, mix with QS using alpha,
    optionally filter by threshold, and return top-k recommendations.
    """
    if isinstance(student_row, dict):
        student_row = pd.Series(student_row)

    # 1) candidates
    df_candidates = build_candidate_catalog(
        df_qs,
        degree=degree,
        country=country,
        uni_col=uni_col,
        degree_col=degree_col,
        country_col=country_col,
        qs_col=qs_col,
    )
    if df_candidates.empty:
        return df_candidates

    # 2) normalize QS
    df_candidates = add_qs_normalized(df_candidates, qs_col=qs_col, degree_col=degree_col)

    # 3) build scoring frame
    df_score = df_candidates.copy()
    for c in feature_cols:
        if c not in df_score.columns:
            df_score[c] = student_row.get(c, None)

    # 4) predict probabilities
    df_score["p_admit"] = model.predict_proba(df_score[feature_cols])[:, 1]

    # 5) optional threshold filter (objective A)
    if threshold is not None:
        df_score = df_score[df_score["p_admit"] >= threshold].copy()

    # 6) mix
    df_score["final_score"] = alpha * df_score["p_admit"] + (1 - alpha) * df_score["qs_norm"]

    # 7) rank
    df_score = df_score.sort_values("final_score", ascending=False).head(k)

    # Keep nice columns for display
    keep = [uni_col, country_col, degree_col, qs_col, "qs_norm", "p_admit", "final_score"]
    keep = [c for c in keep if c in df_score.columns]
    return df_score[keep].reset_index(drop=True)