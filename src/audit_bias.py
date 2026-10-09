"""Step 3: run from src/ to audit saved validation predictions only."""
from pathlib import Path
import hashlib
import json
from importlib.metadata import version

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from fairlearn.metrics import (MetricFrame, count, selection_rate,
                              demographic_parity_difference, equalized_odds_difference)

ROOT = Path(__file__).resolve().parents[1]
ATTRIBUTES = ["age_group", "gender_codebook_proxy"]


def positives(y_true, y_pred):
    return int(np.sum(np.asarray(y_true) == 1))


def negatives(y_true, y_pred):
    return int(np.sum(np.asarray(y_true) == 0))


def tpr(y_true, y_pred):
    # Undefined rates stay NaN rather than misleadingly becoming zero.
    mask = np.asarray(y_true) == 1
    return float(np.mean(np.asarray(y_pred)[mask] == 1)) if mask.any() else np.nan


def fpr(y_true, y_pred):
    mask = np.asarray(y_true) == 0
    return float(np.mean(np.asarray(y_pred)[mask] == 1)) if mask.any() else np.nan


def main():
    source = ROOT / "reports/baseline_validation_predictions.csv"
    data = pd.read_csv(source)
    required = ["row_id", "y_true", "y_pred", "approval_score", *ATTRIBUTES]
    if data[required].isna().any().any() or data.row_id.duplicated().any():
        raise ValueError("Missing audit values or duplicate row IDs.")
    for column in ["y_true", "y_pred"]:
        if not set(data[column]).issubset({0, 1}):
            raise ValueError(f"{column} must contain binary labels.")
    metadata = json.loads((ROOT / "models/baseline_metadata.json").read_text())
    if not data.approval_score.between(0, 1).all():
        raise ValueError("Scores must be between zero and one.")
    assert np.array_equal(data.y_pred, (data.approval_score >= metadata["threshold"]).astype(int))
    # Confirm audit rows and labels belong to the reserved validation split.
    validation = pd.read_csv(ROOT / "data/splits/validation.csv").set_index("row_id")
    if set(data.row_id) != set(validation.index):
        raise ValueError("Audit row IDs do not match the validation split.")
    aligned = validation.loc[data.row_id]
    assert np.array_equal(data.y_true, aligned["loan_approved"])
    for attribute in ATTRIBUTES:
        assert np.array_equal(data[attribute], aligned[attribute])

    metrics = {"count": count, "positive_count": positives,
               "negative_count": negatives, "selection_rate": selection_rate,
               "accuracy": accuracy_score, "true_positive_rate": tpr,
               "false_positive_rate": fpr}
    summaries, tables = [], []
    report_dir = ROOT / "reports/fairness"
    report_dir.mkdir(parents=True, exist_ok=True)
    for attribute in ATTRIBUTES:
        frame = MetricFrame(metrics=metrics, y_true=data.y_true,
                            y_pred=data.y_pred, sensitive_features=data[attribute])
        table = frame.by_group.rename_axis("group").reset_index()
        if attribute == "age_group":
            table = table.set_index("group").reindex(["<25", "25-39", "40+"]).reset_index()
        for column in ["count", "positive_count", "negative_count"]:
            table[column] = table[column].astype(int)
        table.insert(0, "attribute", attribute)
        table["label_positive_rate"] = table.positive_count / table["count"]
        # A descriptive flag, NOT a significance test or universal cutoff.
        table["small_denominator"] = (table.positive_count < 20) | (table.negative_count < 20)
        table["interpretation"] = (
            "unverified codebook grouping; demonstration only" if attribute.startswith("gender")
            else "descriptive validation comparison; not a causal finding")
        dp = demographic_parity_difference(data.y_true, data.y_pred,
                                           sensitive_features=data[attribute])
        defined = (table.positive_count > 0).all() and (table.negative_count > 0).all()
        eo = equalized_odds_difference(data.y_true, data.y_pred,
                                      sensitive_features=data[attribute],
                                      agg="worst_case") if defined else np.nan
        tpr_gap = table.true_positive_rate.max() - table.true_positive_rate.min() if (table.positive_count > 0).all() else np.nan
        fpr_gap = table.false_positive_rate.max() - table.false_positive_rate.min() if (table.negative_count > 0).all() else np.nan
        # Independently verify the Fairlearn gap calculations.
        assert np.isclose(dp, table.selection_rate.max() - table.selection_rate.min())
        if defined:
            assert np.isclose(eo, max(tpr_gap, fpr_gap))
        summaries.append({"attribute": attribute, "split": "validation",
                          "threshold": metadata["threshold"], "n": len(data),
                          "demographic_parity_difference": dp,
                          "true_positive_rate_difference": tpr_gap,
                          "false_positive_rate_difference": fpr_gap,
                          "equalized_odds_difference": eo})
        tables.append(table)
        table.to_csv(report_dir / f"{attribute}_metrics.csv", index=False)
        plot_data = table.melt(id_vars=["group", "count", "positive_count", "negative_count"],
                               value_vars=["selection_rate", "true_positive_rate", "false_positive_rate"],
                               var_name="metric", value_name="rate")
        title = "Age groups" if attribute == "age_group" else "Unverified gender codebook groups (demo only)"
        fig = px.bar(plot_data, x="group", y="rate", color="metric", barmode="group",
                     hover_data=["count", "positive_count", "negative_count"],
                     title=f"{title} | validation n={len(data)}", range_y=[0, 1])
        fig.update_layout(yaxis_tickformat=".0%", xaxis_title="Group", template="plotly_white")
        fig.write_html(report_dir / f"{attribute}_comparison.html", include_plotlyjs=True)

    all_groups = pd.concat(tables, ignore_index=True)
    all_groups.to_csv(report_dir / "group_metrics.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(report_dir / "fairness_summary.csv", index=False)
    pd.Series(frame.overall, name="value").rename_axis("metric").to_csv(report_dir / "overall_metrics.csv")
    # Static companion for sharing without running Streamlit.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, table, title in zip(axes, tables, ["Age groups", "Unverified gender codebook groups"]):
        x = np.arange(len(table))
        for offset, (column, label, color) in enumerate([
            ("selection_rate", "Approval proxy rate", "#2563eb"),
            ("true_positive_rate", "True-positive rate", "#0d9488"),
            ("false_positive_rate", "False-positive rate", "#d97706")]):
            ax.bar(x + (offset - 1) * .25, table[column], width=.25, label=label, color=color)
        ax.set_xticks(x, [f"{r.group}\nn={r.count}; positives={r.positive_count}\nnegatives={r.negative_count}" for r in table.itertuples()])
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("Rate")
        ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Baseline validation audit | threshold 0.5\nDescriptive point estimates; gender mapping is unverified", fontsize=12)
    fig.savefig(report_dir / "baseline_group_rates.png", dpi=160)
    plt.close(fig)
    notes = {
        "prediction_file_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "fairlearn_version": version("fairlearn"), "split": "validation",
        "limitations": [
            "Approval labels are good-credit-risk proxies, not observed approvals.",
            "Gender codebook grouping is unverified; no real gender-bias claims are justified.",
            "Point estimates only: no confidence intervals or significance tests.",
            "Small denominators can cause large changes in rates; <20 is a descriptive flag only.",
            "Separate attribute audits can hide intersectional disparities.",
            "Parity is not proof of fairness; this audit does not establish causation.",
            "Demographic parity and equalized odds encode different goals and may conflict.",
        ],
        "metric_reference": "https://fairlearn.org/v0.13/api_reference/generated/fairlearn.metrics.equalized_odds_difference.html",
    }
    (report_dir / "audit_metadata.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print(all_groups.drop(columns="interpretation").round(4).to_string(index=False))
    print("\nFairness gaps (multiply by 100 for percentage points):")
    print(summary.round(4).to_string(index=False))
    print(f"\nSaved tables, charts, and limitations to {report_dir}")
    print("Validation alignment and metric checks passed. No test evaluation or retraining.")


if __name__ == "__main__":
    main()
