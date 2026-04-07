import pandas as pd

def add_student_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a stable student_id by hashing student-level attributes.
    Avoids university/degree/admitted/qs_score to prevent leakage.
    """
    df = df.copy()

    id_cols = [
        "school_curriculum",
        "gpa",
        "toefl",
        "sat",
        "profile_score",
        "peak",
        "school_grade",
    ]
    id_cols = [c for c in id_cols if c in df.columns]

    id_matrix = (
        df[id_cols]
        .fillna("")
        .astype(str)
        .apply(lambda s: s.str.strip().str.lower())
    )

    df["student_id"] = pd.util.hash_pandas_object(id_matrix, index=False).astype("int64").astype(str)
    return df