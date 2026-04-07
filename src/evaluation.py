from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, classification_report
from sklearn.model_selection import RandomizedSearchCV, GroupKFold

def evaluate_model(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "f1": f1_score(y_test, pred),
        "report": classification_report(y_test, pred, digits=3),
    }