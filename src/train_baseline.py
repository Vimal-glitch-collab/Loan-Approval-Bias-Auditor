"""Copy into your project's src/ folder; run python src/train_baseline.py."""
from pathlib import Path
import hashlib
import json
import platform
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
TARGET = "loan_approved"
PROTECTED = ["gender_codebook_proxy", "age_group"]
SEED = 42
THRESHOLD = 0.5


def main():
    source = ROOT / "data/german_credit_clean.csv"
    if not source.exists():
        raise FileNotFoundError(
            f"Missing {source}. Put this script in src/ and run load_data.py first.")
    df = pd.read_csv(source)
    required = [TARGET, *PROTECTED, "age", "personal_status_sex"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError("Target must contain only 0 and 1, with both present.")
    if df[PROTECTED].isna().any().any():
        raise ValueError("Audit attributes must not be missing.")

    # Row IDs preserve alignment across features, labels, and audit attributes.
    df.index.name = "row_id"
    train_val, test = train_test_split(
        df, test_size=0.20, stratify=df[TARGET], random_state=SEED)
    train, validation = train_test_split(
        train_val, test_size=0.25, stratify=train_val[TARGET], random_state=SEED)
    splits = {"train": train, "validation": validation, "test": test}
    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(validation.index).isdisjoint(test.index)

    # Exclude direct protected attributes and any legacy copy of the target.
    excluded = {*required, "credit_risk_code", "row_id"}
    features = [column for column in df.columns if column not in excluded]
    X_train, y_train = train[features], train[TARGET]
    numeric = X_train.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in features if column not in numeric]

    # Learn imputation values and categories from training data only.
    # Dense output is small here and convenient for later SHAP explanations.
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    model = Pipeline([
        ("preprocess", preprocessing),
        ("classifier", XGBClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", random_state=SEED, n_jobs=2,
        )),
    ])
    model.fit(X_train, y_train)
    # A 70% majority class makes accuracy alone misleading.
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    for folder in ["models", "reports", "data/splits"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(ROOT / f"data/splits/{name}.csv", index=True)

    rows = []
    # Inspect training fit and validation generalization; leave test untouched.
    for name, frame in [("train", train), ("validation", validation)]:
        for label, estimator in [("xgboost", model), ("majority_dummy", dummy)]:
            positive_index = list(estimator.classes_).index(1)
            probability = estimator.predict_proba(frame[features])[:, positive_index]
            prediction = (probability >= THRESHOLD).astype(int)
            actual = frame[TARGET]
            tn, fp, fn, tp = confusion_matrix(actual, prediction, labels=[0, 1]).ravel()
            rows.append({
                "model": label, "split": name, "n": len(frame),
                "threshold": THRESHOLD,
                "accuracy": accuracy_score(actual, prediction),
                "balanced_accuracy": balanced_accuracy_score(actual, prediction),
                "precision": precision_score(actual, prediction, zero_division=0),
                "recall": recall_score(actual, prediction, zero_division=0),
                "f1": f1_score(actual, prediction, zero_division=0),
                "roc_auc": roc_auc_score(actual, probability),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            })
            if label == "xgboost":
                # Save aligned audit inputs for Step 3 and the dashboard.
                audit = frame[PROTECTED].copy()
                audit["y_true"] = actual
                audit["approval_score"] = probability
                audit["y_pred"] = prediction
                audit.to_csv(ROOT / f"reports/baseline_{name}_predictions.csv")

    model_path = ROOT / "models/baseline_pipeline.joblib"
    joblib.dump(model, model_path)
    joblib.dump(dummy, ROOT / "models/majority_dummy.joblib")
    # Verify the saved pipeline reproduces in-memory predictions.
    reloaded = joblib.load(model_path)
    np.testing.assert_allclose(
        model.predict_proba(validation[features]),
        reloaded.predict_proba(validation[features]),
    )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(ROOT / "reports/baseline_metrics.csv", index=False)
    metadata = {
        "random_state": SEED, "threshold": THRESHOLD,
        "feature_columns": features, "numeric_columns": numeric,
        "categorical_columns": categorical, "protected_attributes": PROTECTED,
        "target": TARGET, "split_sizes": {k: len(v) for k, v in splits.items()},
        "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "python_version": platform.python_version(),
        "versions": {p: version(p) for p in ["pandas", "numpy", "scikit-learn", "xgboost", "joblib"]},
        "limitations": [
            "Approval is a good-credit-risk proxy, not an observed approval decision.",
            "Gender codebook mapping is disputed and cannot support real gender-bias claims.",
            "Excluding protected attributes does not ensure fairness.",
            "The 0.5 threshold is a demo default; scores are not calibrated probabilities.",
            "Test set is reserved; do not use it for model or mitigation selection.",
        ],
    }
    (ROOT / "models/baseline_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print("Split sizes:", metadata["split_sizes"])
    print("Predictor count:", len(features))
    print(metrics.round(4).to_string(index=False))
    print("Saved pipeline, metrics, aligned predictions, metadata, and splits.")
    print("Saved-pipeline prediction check passed. Test set remains unevaluated.")


if __name__ == "__main__":
    main()
