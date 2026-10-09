"""Steps 4–5: training-only reweighting, frozen evaluation, and cached SHAP."""
from pathlib import Path
import hashlib
import json
from importlib.metadata import version
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, confusion_matrix)
from fairlearn.metrics import MetricFrame, selection_rate, count
from audit_bias import tpr, fpr, positives, negatives

ROOT = Path(__file__).resolve().parents[1]
LABEL = "loan_approved"
ATTRIBUTES = ["age_group", "gender_codebook_proxy"]


def wilson(successes, total):
    """Approximate 95% binomial intervals; not significance tests for gaps."""
    if total == 0:
        return np.nan, np.nan
    z = 1.96
    p = successes / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/(1 + z*z/total)
    return max(0, center-half), min(1, center+half)


def evaluate(model, name, split_name, data, features):
    score = model.predict_proba(data[features])[:, 1]
    prediction = (score >= .5).astype(int)
    y = data[LABEL].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    overall = dict(model=name, split=split_name, n=len(data), threshold=.5,
                   accuracy=accuracy_score(y, prediction),
                   balanced_accuracy=balanced_accuracy_score(y, prediction),
                   precision=precision_score(y, prediction, zero_division=0),
                   recall=recall_score(y, prediction, zero_division=0),
                   f1=f1_score(y, prediction, zero_division=0),
                   roc_auc=roc_auc_score(y, score), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    predictions = data[["row_id", *ATTRIBUTES]].copy()
    predictions["model"], predictions["split"] = name, split_name
    predictions["y_true"], predictions["score"], predictions["y_pred"] = y, score, prediction
    groups, gaps = [], []
    for attribute in ATTRIBUTES:
        frame = MetricFrame(metrics={"count": count, "positive_count": positives,
            "negative_count": negatives, "selection_rate": selection_rate,
            "accuracy": accuracy_score, "true_positive_rate": tpr, "false_positive_rate": fpr},
            y_true=y, y_pred=prediction, sensitive_features=data[attribute])
        table = frame.by_group.rename_axis("group").reset_index()
        table["attribute"], table["model"], table["split"] = attribute, name, split_name
        for col in ["count", "positive_count", "negative_count"]:
            table[col] = table[col].astype(int)
        table["small_denominator"] = (table.positive_count < 20) | (table.negative_count < 20)
        for rate, denom in [("selection_rate", "count"), ("true_positive_rate", "positive_count"), ("false_positive_rate", "negative_count")]:
            intervals = [wilson(round(r[rate]*r[denom]) if r[denom] else 0, r[denom]) for _, r in table.iterrows()]
            table[rate+"_low"], table[rate+"_high"] = zip(*intervals)
        dp = table.selection_rate.max() - table.selection_rate.min()
        tg = table.true_positive_rate.max() - table.true_positive_rate.min() if (table.positive_count > 0).all() else np.nan
        fg = table.false_positive_rate.max() - table.false_positive_rate.min() if (table.negative_count > 0).all() else np.nan
        eo = max(tg, fg) if np.isfinite([tg, fg]).all() else np.nan
        gaps.append(dict(model=name, split=split_name, attribute=attribute,
                         demographic_parity_difference=dp, equalized_odds_difference=eo,
                         true_positive_rate_difference=tg, false_positive_rate_difference=fg))
        groups.append(table)
    return overall, predictions, pd.concat(groups), pd.DataFrame(gaps)


def cache_shap(models, splits, features, output):
    import shap
    all_importance, all_local = [], []
    for name, model in models.items():
        prep, classifier = model.named_steps["preprocess"], model.named_steps["classifier"]
        explainer = shap.TreeExplainer(classifier, model_output="raw", feature_perturbation="tree_path_dependent")
        # Map each one-hot component back to its original input feature.
        numeric = list(prep.transformers_[0][2])
        categorical = list(prep.transformers_[1][2])
        categories = prep.named_transformers_["categorical"].named_steps["encoder"].categories_
        origins = numeric + [col for col, values in zip(categorical, categories) for _ in values]
        for split, frame in splits.items():
            transformed = prep.transform(frame[features])
            explanation = explainer(transformed)
            raw = np.asarray(explanation.values)
            base = np.broadcast_to(explanation.base_values, (len(frame),)).astype(float)
            assert raw.shape == (len(frame), len(origins))
            grouped = np.column_stack([raw[:, np.array(origins) == col].sum(axis=1) for col in features])
            # Additivity: base + contributions gives margin; sigmoid gives model score.
            reconstructed = 1 / (1 + np.exp(-(base + grouped.sum(axis=1))))
            np.testing.assert_allclose(reconstructed, model.predict_proba(frame[features])[:, 1], atol=2e-6)
            local = pd.DataFrame(grouped, columns=features)
            local.insert(0, "row_id", frame.row_id.to_numpy())
            local.insert(0, "split", split)
            local.insert(0, "model", name)
            local["base_value"] = base
            all_local.append(local)
            importance = pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(grouped).mean(axis=0)})
            importance["model"], importance["split"] = name, split
            all_importance.append(importance)
        joblib.dump(explainer, ROOT / f"models/{name}_explainer.joblib")
        print(f"SHAP cached and additivity verified: {name}", flush=True)
    pd.concat(all_local).to_csv(output / "shap_local.csv", index=False)
    pd.concat(all_importance).to_csv(output / "shap_global.csv", index=False)


def main():
    meta = json.loads((ROOT / "models/baseline_metadata.json").read_text())
    features = meta["feature_columns"]
    train = pd.read_csv(ROOT / "data/splits/train.csv")
    validation = pd.read_csv(ROOT / "data/splits/validation.csv")
    baseline = joblib.load(ROOT / "models/baseline_pipeline.joblib")
    clean = ROOT / "data/german_credit_clean.csv"
    assert hashlib.sha256(clean.read_bytes()).hexdigest() == meta["data_sha256"], "Dataset changed; rebuild Step 2."
    assert set(train.row_id).isdisjoint(validation.row_id)
    assert not {LABEL, "age", "age_group", "gender_codebook_proxy", "personal_status_sex"}.intersection(features)
    output = ROOT / "reports/final"
    output.mkdir(parents=True, exist_ok=True)

    # Independence weights: P(age_group) * P(label) / P(age_group, label).
    # No validation/test labels enter weights or the fitted model.
    joint = pd.crosstab(train.age_group, train[LABEL])
    if (joint == 0).any().any():
        raise ValueError("Every age/label cell needs training records for reweighting.")
    expected = np.outer(joint.sum(axis=1), joint.sum(axis=0)) / len(train)
    weight_table = pd.DataFrame(expected / joint.to_numpy(), index=joint.index, columns=joint.columns)
    weights = np.array([weight_table.loc[g, y] for g, y in zip(train.age_group, train[LABEL])])
    np.testing.assert_allclose(weights.mean(), 1.)
    weight_rows = []
    for group in joint.index:
        for label in joint.columns:
            mask = (train.age_group == group) & (train[LABEL] == label)
            np.testing.assert_allclose(weights[mask].sum(), expected[list(joint.index).index(group), list(joint.columns).index(label)])
            weight_rows.append(dict(age_group=group, label=int(label), count=int(joint.loc[group, label]), weight=float(weight_table.loc[group, label])))
    pd.DataFrame(weight_rows).to_csv(output / "training_weights.csv", index=False)
    mitigated = clone(baseline)
    mitigated.fit(train[features], train[LABEL], classifier__sample_weight=weights)
    joblib.dump(mitigated, ROOT / "models/reweighted_pipeline.joblib")
    reloaded = joblib.load(ROOT / "models/reweighted_pipeline.joblib")
    np.testing.assert_allclose(reloaded.predict_proba(validation[features]), mitigated.predict_proba(validation[features]))
    models = {"Baseline": baseline, "Reweighted": mitigated}
    results = [evaluate(m, n, "validation", validation, features) for n, m in models.items()]
    print("Step 4 validation results:", flush=True)
    print(pd.DataFrame([r[0] for r in results]).round(4).to_string(index=False), flush=True)
    # Freeze design BEFORE evaluating the test set. Both candidates are reported.
    manifest = {"method": "Age × label independence reweighting", "fit_split": "train (600)",
        "threshold": .5, "candidate_count": 1, "selection": "Both fixed candidates reported; no test-based tuning",
        "features": features, "data_sha256": meta["data_sha256"],
        "versions": {p: version(p) for p in ["pandas", "numpy", "scikit-learn", "xgboost", "fairlearn", "shap", "streamlit", "plotly"]},
        "shap_units": "Raw XGBoost margin (log-odds); one-hot contributions summed per original feature",
        "limitations": meta["limitations"][:4] + [
            "Reweighting does not guarantee demographic parity or equalized odds on unseen data.",
            "Test has only 200 records. Wilson intervals are approximate group-rate intervals, not gap significance tests.",
            "SHAP explains model associations, not causal effects or proof of fairness.",
            "Age groups are arbitrary; separate attribute audits can hide intersectional disparities.",
            "Baseline Step 2 test reservation is now superseded by this final evaluation."]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    test = pd.read_csv(ROOT / "data/splits/test.csv")
    assert set(test.row_id).isdisjoint(train.row_id) and set(test.row_id).isdisjoint(validation.row_id)
    results += [evaluate(m, n, "test", test, features) for n, m in models.items()]
    pd.DataFrame([r[0] for r in results]).to_csv(output / "performance.csv", index=False)
    for i, file in [(1, "predictions.csv"), (2, "group_metrics.csv"), (3, "fairness.csv")]:
        pd.concat([r[i] for r in results], ignore_index=True).to_csv(output / file, index=False)
    pd.concat([validation.assign(split="validation"), test.assign(split="test")]).to_csv(output / "cases.csv", index=False)
    print("Frozen final test results:", flush=True)
    print(pd.DataFrame([r[0] for r in results if r[0]["split"] == "test"]).round(4).to_string(index=False), flush=True)
    cache_shap(models, {"validation": validation, "test": test}, features, output)
    print("Steps 4–5 complete. Models, metrics, cases, and SHAP are saved.", flush=True)


if __name__ == "__main__":
    main()
