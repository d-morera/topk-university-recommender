# 🎓 Top-K University Recommender — University Admission Predictor & Top-K Recommender

> *Given a student's academic profile and intended major, which universities should they actually apply to?*

Top-K University Recommender is an end-to-end ML system that predicts a student's probability of university admission and uses that prediction to generate personalized, ranked recommendations — balancing **feasibility** (how likely is admission?) against **prestige** (how good is the university?).

Built as a realistic portfolio project for a <u>real admissions consultancy<u>. Real data from the organization stays private; the pipeline, modeling decisions, and recommender logic are fully reproducible.

---

## What it does

- **Predicts admission probability** — `P(admitted=1 | student profile, university, degree, QS score, ...)` using a calibrated Random Forest with threshold tuning optimized to minimize false positives.
- **Recommends Top-K universities** — ranks candidates from the QS World University Rankings by a configurable blend of admission probability and normalized prestige score.
- **Filters by degree and country** — narrows the candidate pool to what's actually relevant for each student.
- **Handles the hard data problems** — multi-label degree mapping, group-aware train/test splitting to prevent student leakage, deterministic university name normalization, and QS tail imputation.

---

## How it works

```
Raw Data (Excel)
    │
    ├─ data/raw/Five Lands Stats.xlsx      ← student profiles and admission info
    └─ data/raw/University Ranking by Major 2025.xlsx  ← QS rankings per major
    │
    ▼
[src/cleaning_students.py]   Standardize schema, coerce types, drop noise columns
[src/cleaning_qs.py]         Read multi-sheet QS Excel, augment tail scores, rename cols
[src/merge_students_qs.py]   Normalize university names + map degrees → QS categories
[src/features.py]            Hash-based student_id (group split key)
    │
    ▼
[src/modeling.py]            ColumnTransformer → RandomForest pipeline
                             GroupKFold CV + RandomizedSearchCV
                             OOF threshold tuning (precision-conservative)
    │
    ▼
[models/admission_prob_model.joblib]   Saved model bundle (model + threshold + feature_cols)
[reports/tables/qs_catalog.csv]        Exported QS candidate catalog
    │
    ▼
[src/recommender.py]         Filter catalog → normalize QS per degree → score with model
                             Apply threshold filter → rank by final_score → return Top-K

final_score = α · P(admit) + (1−α) · QS_norm
```

The full walkthrough lives in [`notebooks/00_main_end_to_end.ipynb`](notebooks/00_main_end_to_end.ipynb).

---

## Data

> ⚠️ **Real data is private and not included in this repository.**

The pipeline expects two Excel files in `data/raw/`:

| File | Description |
|------|-------------|
| `data/raw/Five Lands Stats.xlsx` | Student application records (one row per student–university pair). Contains academic profile features and admission outcome. |
| `data/raw/University Ranking by Major 2025.xlsx` | QS World University Rankings 2025, one sheet per subject/major. |

Both files are excluded via `.gitignore` (`data/raw/`, `data/processed/`, `*.xlsx`). If you want to reproduce the pipeline, place your own data files at these paths following the column schema described in [`src/cleaning_students.py`](src/cleaning_students.py).

A sample schema is available in `data/sample/` for reference.

---

## Setup

**Requirements:** Python 3.10+ (developed on 3.14), Jupyter Lab.

```bash
# 1) Clone the repo
git clone https://github.com/d-morera/topk-university-recommender.git
cd topk-university-recommender

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

Key libraries: `scikit-learn`, `pandas`, `numpy`, `openpyxl`, `joblib`, `matplotlib`.

---

## Run

Open and run the main notebook end-to-end:

```bash
jupyter lab notebooks/00_main_end_to_end.ipynb
```

The notebook is organized into sequential sections:

1. **Data loading & cleaning** — reads both Excel sources, standardizes schemas
2. **Degree mapping & merge** — maps student degrees to QS categories, merges datasets
3. **Feature engineering** — generates `student_id`, encodes features, prepares `X` and `y`
4. **Model training** — builds the preprocessing pipeline and fits the Random Forest
5. **Threshold tuning** — selects a conservative threshold from OOF predictions
6. **Model export** — saves the bundle to `models/admission_prob_model.joblib`
7. **Top-K recommendations** — runs the recommender on sample students in `src/app.py`

---

## Modeling choices

### Group-aware splitting — no student leakage

Each student has multiple application rows (one per university). A naive row-level train/test split would put the same student in both sets, inflating metrics. To prevent this:

- A **stable `student_id`** is generated in [`src/features.py`](src/features.py) by hashing the student's academic profile columns (`gpa`, `toefl`, `sat`, `profile_score`, `school_curriculum`, etc.) — deliberately excluding `university`, `degree`, `admitted`, and `qs_score` to avoid leakage.
- **`GroupShuffleSplit`** is used for the train/test split.
- **`GroupKFold`** (5 folds) is used for cross-validation and hyperparameter search.

This is the most important data integrity decision in the project. Without it, reported metrics are artificially inflated.

### Random Forest + threshold tuning

[`src/modeling.py`](src/modeling.py) builds a `sklearn` Pipeline:

```
ColumnTransformer
  ├─ Numerical: SimpleImputer(median) → StandardScaler
  └─ Categorical: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')
RandomForestClassifier(n_estimators=600, class_weight='balanced_subsample', ...)
```

Hyperparameters are tuned via `RandomizedSearchCV` with `GroupKFold`, scoring on `average_precision` (PR-AUC), which is more informative than ROC-AUC under class imbalance.

**The threshold is not a model hyperparameter.** It is determined post-training from out-of-fold (OOF) predictions under a conservative policy: the threshold is set to achieve `precision ≥ 0.85` while keeping recall at a reasonable level (notebook section *"Threshold tuning"*). The goal is **Objective A: minimize false positives** — the system should not confidently recommend a university where the student is likely to be rejected.

The effect of this policy is visible in the evaluation results. The final model (M2, without `university`) uses a threshold of **0.82**, which is notably more conservative than the baseline (M1, threshold 0.62):

| Model | Threshold | ROC-AUC | PR-AUC | Precision (class 1) | FP count |
|-------|-----------|---------|--------|---------------------|----------|
| M1 — with `university` | 0.62 | 0.762 | 0.893 | 0.831 | 13 |
| M2 — without `university` | 0.82 | 0.764 | 0.882 | **0.897** | **6** |

M2 cuts false positives in half (13 → 6) at the cost of more false negatives — which is exactly the trade-off we want: it is better to miss a reachable university than to recommend one where admission is unlikely. The ROC-AUC and PR-AUC remain essentially equivalent between the two models, meaning the drop in FP comes from the stricter threshold policy, not from a weaker model.

The threshold is saved as part of the model bundle alongside `feature_cols` so inference is self-contained.

### Ablation: with vs. without `university`

`university` is a high-cardinality categorical variable. With limited training data, including it risks **memorization**: the model learns "this specific university tends to admit" rather than learning genuine student-level patterns.

An ablation study (notebook section *"Ablation"*) compares the two models under their respective conservative thresholds on a held-out test set of 115 samples (33 rejected, 82 admitted):

| Setting | Threshold | FP rate | FP count | Precision (class 1) | Recall (class 1) |
|---------|-----------|---------|----------|---------------------|------------------|
| M1 — with `university` | 0.62 | 0.39 | 13 / 33 | 0.831 | 0.780 |
| M2 — without `university` | 0.82 | **0.18** | **6 / 33** | **0.897** | 0.634 |

Removing `university` roughly **halves the false positive rate** (0.39 → 0.18) under the conservative threshold policy. The trade-off is a lower recall on admitted students (0.780 → 0.634): the model becomes more conservative overall and misses some viable universities. For the recommendation use case, this is the right trade-off — we prioritize **not recommending universities with low admission likelihood** over maximizing coverage.

**The model without `university` (M2) is the default for recommendations.** Since the recommender scores universities from the QS catalog — many of which were never seen in training — generalization matters more than in-sample fit.

---

## Top-K recommender

The recommender lives in [`src/recommender.py`](src/recommender.py) and is invoked via `recommend_top_k()`.

### Candidate catalog

The QS dataset is exported to [`reports/tables/qs_catalog.csv`](reports/tables/qs_catalog.csv) with columns `university`, `country`, `degree`, `qs_score`. It is filtered by:
- **`degree`** (mapped QS category, e.g. `"Computer Science"`, `"Business"`)
- **`country`** (optional; e.g. `"UNITED KINGDOM"`, `"UNITED STATES"`)

### QS normalization

QS scores are normalized **within each degree category** to make them comparable to admission probabilities (both in [0, 1]):

```
QS_norm = (QS - QS_min) / (QS_max - QS_min)   [per degree]
```

This is critical: a QS score of 80 means something very different in Engineering than in Music. Per-degree normalization makes `α` interpretable as a true safety-vs-prestige weight.

### Scoring formula

For each candidate university, the final ranking score is:

```
final_score = α · P(admit) + (1 − α) · QS_norm
```

- `α = 1.0` → pure safety (rank by admission probability only)
- `α = 0.0` → pure prestige (rank by QS score only)
- `α = 0.7` → default: prioritize feasibility, reward prestige

An optional `threshold` parameter pre-filters candidates where `P(admit) < threshold` before scoring, implementing a hard safety floor (Objective A).

### Usage example

```python
from src.recommender import recommend_top_k
import joblib, pandas as pd

bundle = joblib.load("models/admission_prob_model.joblib")
df_qs  = pd.read_csv("reports/tables/qs_catalog.csv")

student = {
    "school_curriculum": "IB",
    "school_grade": 38,
    "gpa": 3.8,
    "toefl": 105,
    "sat": 1400,
    "profile_score": 14,
    "peak": 1,
    "sat_required": 1,
}

recommendations = recommend_top_k(
    model=bundle["model"],
    student_row=student,
    df_qs=df_qs,
    feature_cols=bundle["feature_cols"],
    degree="Computer Science",
    country="UNITED KINGDOM",
    alpha=0.7,
    k=10,
    threshold=bundle["threshold"],
)
print(recommendations)
```

---

## Outputs & artifacts

| Path | Description |
|------|-------------|
| `models/admission_prob_model.joblib` | Serialized bundle: `{"model": Pipeline, "threshold": float, "feature_cols": list}` |
| `reports/tables/qs_catalog.csv` | Cleaned QS catalog used as the recommendation candidate pool |
| `reports/figures/` | Evaluation plots (precision-recall curve, threshold sweep, alpha sweep) |

> Note: `models/*.joblib` and `*.csv` are excluded from the repo by `.gitignore`. They are generated locally when running the main notebook.

---

## Repository structure

```
FLRecomendator/
├── notebooks/
│   ├── 00_main_end_to_end.ipynb   ← Main notebook: full pipeline
│   └── archive/                   ← Older versions (not maintained) - 2024
│       ├── ML_Probabilistico.ipynb
│       ├── ML_TopK.ipynb
│       ├── ML_TratamientoDatos.ipynb
│       ├── ML_dataAugmentation.ipynb
│       └── ML_dataExpansion.ipynb
│
├── src/
│   ├── config.py                  ← Project-root-relative path constants
│   ├── cleaning_students.py       ← Schema standardization + type coercion
│   ├── cleaning_qs.py             ← QS multi-sheet reader + tail augmentation
│   ├── merge_students_qs.py       ← University name normalization + degree mapping
│   ├── features.py                ← student_id hash (group split key)
│   ├── modeling.py                ← Preprocessor, RF pipeline, RandomizedSearchCV
│   ├── evaluation.py              ← ROC-AUC, PR-AUC, classification report
│   ├── recommender.py             ← Top-K logic: filter, QS norm, score, rank
│   ├── utils.py                   ← Safe Excel loaders
│   └── app.py                     ← (WIP) Lightweight demo interface
│
├── data/
│   ├── raw/                       ← 🔒 Private — not in repo
│   ├── processed/                 ← 🔒 Private — not in repo
│   └── sample/                    ← Schema reference (no real data)
│
├── models/
    └── admisssion_prob_model.joblib   ← Generated locally — not in repo
├── reports/
│   ├── figures/
│   └── tables/
│       └── qs_catalog.csv         ← Generated locally — not in repo
│
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Limitations & next steps

**Current limitations:**
- **QS coverage gap:** Some universities in the student dataset have no QS ranking entry. After name normalization, unmatched universities receive `qs_score = NaN` and are excluded from recommendations. No scores are imputed to avoid introducing noise.
- **Dataset size:** The model is trained on a relatively small, consultancy-specific dataset. Metrics should be interpreted with this in mind — generalization to other student populations is untested.
- **Static catalog:** The QS catalog used is from 2025 and is not updated automatically.
- **Single model family:** Only Random Forest has been tuned. Gradient boosting (XGBoost, LightGBM) could offer better calibrated probabilities.

**Potential next steps:**
- [ ] Add gradient boosting baseline and compare calibration
- [ ] Calibrate output probabilities with Platt scaling or isotonic regression
- [ ] Build an interactive Streamlit/Gradio demo around `src/app.py`
- [ ] Automate QS catalog refresh from QS API or scraper
- [ ] Expand university name normalization coverage to reduce NaN rate
- [ ] Experiment with a listwise learning-to-rank objective instead of pointwise scoring

---

## License

[MIT](LICENSE) — see the LICENSE file for details.

---

*Built with Python 3.14 · scikit-learn 1.8 · pandas 3.0 · JupyterLab 4.5*