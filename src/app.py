import joblib
import pandas as pd
from pathlib import Path
from recommender import recommend_top_k

def main():
    root = Path(__file__).resolve().parents[1]
    bundle = joblib.load(root / "models" / "admission_model_no_university.joblib")
    model = bundle["model"]
    threshold = bundle["threshold"]
    feature_cols = bundle["feature_cols"]

    df_qs_catalog = pd.read_csv(root / "reports" / "tables" / "qs_catalog.csv")

    student = {
        "school_curriculum": "BASE 7",
        "school_grade": 33,
        "gpa": 3.5,
        "toefl": 102,
        "sat": 1000,
        "profile_score": 9,
        "peak": 0,
        "sat_required": 1,
    }

    degree = "Business"
    country = "SPAIN"

    recs = recommend_top_k(
        model=model,
        student_row=student,
        df_qs_catalog=df_qs_catalog,
        feature_cols=feature_cols,
        degree=degree,
        country=country,
        alpha=0.7,
        k=10,
        threshold=threshold,
    )

    print(recs.to_string(index=False))

if __name__ == "__main__":
    main()
