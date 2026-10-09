"""Smoke-test saved artifacts and every dashboard page without a browser."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = ROOT / "reports/final"
    predictions = pd.read_csv(reports / "predictions.csv")
    shap_values = pd.read_csv(reports / "shap_local.csv")
    manifest = json.loads((reports / "manifest.json").read_text())
    merged = predictions.merge(shap_values, on=["model", "split", "row_id"], validate="one_to_one")
    assert len(merged) == 800
    score = 1/(1+np.exp(-(merged.base_value + merged[manifest["features"]].sum(axis=1))))
    np.testing.assert_allclose(score, merged.score, atol=2e-6)
    np.testing.assert_array_equal(merged.y_pred, (merged.score >= .5).astype(int))
    original = pd.read_csv(ROOT / "reports/baseline_validation_predictions.csv").set_index("row_id")
    baseline = predictions.query("model == 'Baseline' and split == 'validation'").set_index("row_id")
    np.testing.assert_allclose(baseline.loc[original.index, "score"], original.approval_score, atol=1e-7)

    app = AppTest.from_file(str(ROOT / "audit_dashboard.py"), default_timeout=60).run()
    assert not app.exception, str(app.exception)
    for page in ["Fairness audit", "Explain a decision", "Try a scenario", "Project guide", "Overview"]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"{page}: {app.exception}"
        if page == "Fairness audit":
            # Find widgets by their visible labels, independent of page layout.
            next(w for w in app.selectbox if w.label == "Compare groups").set_value("gender_codebook_proxy").run()
            assert not app.exception, str(app.exception)
        if page == "Explain a decision":
            next(w for w in app.selectbox if w.label == "Model to explain").set_value("Reweighted").run()
            assert not app.exception, str(app.exception)
            record = next(w for w in app.selectbox if w.label == "Record to explain")
            record.select_index(1).run()
            assert not app.exception, str(app.exception)
        if page == "Try a scenario":
            app.number_input[0].set_value(5000)
            app.number_input[1].set_value(24)
            next(b for b in app.button if b.label == "Compare model scores").click().run()
            assert not app.exception, str(app.exception)
            assert len(app.metric) == 2
        print(f"Dashboard page passed: {page}", flush=True)
    app.sidebar.selectbox[0].set_value("validation").run()
    assert not app.exception, str(app.exception)
    result = {"artifact_alignment": "passed", "shap_score_reconstruction": "800/800 passed",
              "baseline_validation_preserved": "passed", "dashboard_pages": "5/5 passed",
              "interactive_checks": "gender grouping, reweighted SHAP, case selection, scenario submission, validation selector passed"}
    (reports / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

