from __future__ import annotations

"""Preprocessing + RandomForest baseline + group-aware tuning for admission modeling."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


def preprocessor_schema(X: pd.DataFrame) -> ColumnTransformer:
    """Build numeric/categorical preprocessing for a pandas feature table."""
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(exclude=["number"]).columns

    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
    )


def baseline_model(preprocess: ColumnTransformer, random_state: int = 42) -> Pipeline:
    """Baseline RandomForest pipeline that outputs admission probabilities."""
    clf = RandomForestClassifier(
        n_estimators=600,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=3,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", clf),
        ]
    )


def rf_param_settings() -> dict:
    """Parameter search space used for RandomizedSearchCV tuning."""
    return {
        "model__n_estimators": [300, 600, 1000, 1500],
        "model__max_depth": [None, 5, 10, 20, 40],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 3, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.5, 0.8],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
        "model__bootstrap": [True, False],
    }


def rf_tuning(
    pipe: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups,
    n_iter: int = 40,
    random_state: int = 42,
    scoring: str = "average_precision",
) -> RandomizedSearchCV:
    """Tune the RF pipeline using GroupKFold to avoid student leakage."""
    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError(f"X={len(X)}, y={len(y)}, groups={len(groups)} must match")

    cv = GroupKFold(n_splits=5)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=rf_param_settings(),
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )
    search.fit(X, y, groups=groups)
    return search