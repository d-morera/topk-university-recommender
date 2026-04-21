"""
QS-aware Top-K university recommender.
Combines model admission probabilities with normalised QS prestige scores
to rank candidate universities for a given student profile.
Supports filtering by degree/country and a conservative threshold policy.
"""

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
    Filter the QS catalog by degree and/or country.

    Returns only the columns needed for scoring, deduplicated.
    """
    df = df_qs.copy()

    if degree is not None:
        df = df[df[degree_col] == degree]

    if country is not None:
        df = df[df[country_col] == country]

    cols = [uni_col, country_col, degree_col, qs_col]
    cols = [c for c in cols if c in df.columns]
    return df[cols].drop_duplicates()


def add_qs_normalized(
    df_candidates: pd.DataFrame,
    qs_col: str = "qs_score",
    degree_col: str = "degree",
) -> pd.DataFrame:
    """
    Add a qs_norm column with min-max normalisation computed within each degree.

    Normalising per degree is important because QS score scales differ across
    subjects — it also makes qs_norm compatible in range with p_admit (both 0–1).
    """
    df = df_candidates.copy()

    def _minmax(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mn == mx:
            return pd.Series([0.5] * len(s), index=s.index)  # single entry or all equal → neutral
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
    Replicate student features across all candidates and predict p_admit for each.

    Args:
        model: Trained sklearn pipeline with predict_proba.
        student_row: Single student profile as Series or dict.
        df_candidates: Candidate universities from build_candidate_catalog.
        student_feature_cols: Feature columns the model was trained on.

    Returns:
        df_candidates with a new p_admit column.
    """
    if isinstance(student_row, dict):
        student_row = pd.Series(student_row)

    df = df_candidates.copy()

    # broadcast student features onto every candidate row
    for c in student_feature_cols:
        df[c] = student_row.get(c, None)

    df["p_admit"] = model.predict_proba(
        df[student_feature_cols + [c for c in df_candidates.columns if c not in student_feature_cols]]
    )[:, 1]

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
    Return the Top-K university recommendations for a student profile.

    Scoring formula: final_score = alpha * p_admit + (1 - alpha) * qs_norm
    - alpha=1.0 → rank purely by admission likelihood (safety)
    - alpha=0.0 → rank purely by prestige
    - alpha=0.7 (default) → prioritise feasibility, reward prestige

    Args:
        model: Trained sklearn pipeline with predict_proba.
        student_row: Student profile as Series or dict.
        df_qs: Full QS catalog DataFrame.
        feature_cols: Exact feature columns expected by the model.
        degree: QS subject category to filter on (e.g. "Computer Science").
        country: Country to filter on (optional).
        alpha: Weight for p_admit in the final score (0–1).
        k: Number of recommendations to return.
        threshold: If set, drops candidates with p_admit < threshold before
                   ranking. This is a conservative policy layer, not a model
                   hyperparameter — it directly controls false positive risk.

    Returns:
        DataFrame with top-k rows sorted by final_score descending.
    """
    if isinstance(student_row, dict):
        student_row = pd.Series(student_row)

    # 1) filter candidates
    df_candidates = build_candidate_catalog(
        df_qs, degree=degree, country=country,
        uni_col=uni_col, degree_col=degree_col,
        country_col=country_col, qs_col=qs_col,
    )
    if df_candidates.empty:
        return df_candidates

    # 2) normalise QS within degree so it's on the same 0–1 scale as p_admit
    df_candidates = add_qs_normalized(df_candidates, qs_col=qs_col, degree_col=degree_col)

    # 3) build scoring frame: inject student features the model expects
    df_score = df_candidates.copy()
    for c in feature_cols:
        if c not in df_score.columns:
            df_score[c] = student_row.get(c, None)

    # 4) predict admission probability for every candidate
    df_score["p_admit"] = model.predict_proba(df_score[feature_cols])[:, 1]

    # 5) conservative filter — drop candidates below the admission threshold
    if threshold is not None:
        df_score = df_score[df_score["p_admit"] >= threshold].copy()

    # 6) blend safety and prestige
    df_score["final_score"] = alpha * df_score["p_admit"] + (1 - alpha) * df_score["qs_norm"]

    # 7) rank and return clean output
    df_score = df_score.sort_values("final_score", ascending=False).head(k)

    keep = [uni_col, country_col, degree_col, qs_col, "qs_norm", "p_admit", "final_score"]
    keep = [c for c in keep if c in df_score.columns]
    return df_score[keep].reset_index(drop=True)