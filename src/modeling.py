from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

def preprocessor_schema(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=["number"]).columns
    cat_cols = X.select_dtypes(exclude=["number"]).columns


    num_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
        ])


    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

    return ColumnTransformer([
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ], remainder="drop"
    )

def baseline_model(preprocess: ColumnTransformer, random_state: int = 42) -> Pipeline:
    model = RandomForestClassifier(
        n_estimators=600,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=3,
        max_depth=None,
    )
    return Pipeline([
        ("preprocess", preprocess), 
        ("model", model)
        ])

def rf_param_settings():
    return {
        "model__n_estimators": [300, 600, 1000, 1500],
        "model__max_depth": [None, 5, 10, 20, 40],
        "model__min_samples_split": [2, 5, 10, 20],
        "model__min_samples_leaf": [1, 2, 3, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.5, 0.8],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
        "model__bootstrap": [True, False],
    }

def rf_tuning(pipe, X, y, groups, n_iter=40, random_state=42, scoring="average_precision"):
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