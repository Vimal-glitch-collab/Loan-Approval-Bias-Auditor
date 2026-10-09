> Frontend update: `app.py` now displays only the applicant form and one credit-risk estimate. All 18 inputs are editable. The original audit interface is preserved as `audit_dashboard.py` and can be opened separately with `python -m streamlit run audit_dashboard.py`. Models and audit reports are unchanged. The form uses the baseline because reweighting did not consistently improve fairness. Run `python src/verify_user_frontend.py` to check the new form. The audit walkthrough below describes the preserved interface.
# Loan Approval Bias Auditor

A reproducible credit-risk demonstration with XGBoost, Fairlearn auditing, age-based training reweighting, SHAP explanations, and a Streamlit frontend.

**Scope:** German Credit records contain good/bad credit-risk labels, not actual approval decisions. `loan_approved` is a demonstration proxy. The original sex/status coding is disputed; `gender_codebook_proxy` cannot support claims about actual gender disparities. This is an educational project, not a lending system.

## Run the finished dashboard

In VS Code, open `C:\Users\vimal_z0\Desktop\loan-approval-bias-auditor`, then open Terminal → New Terminal:

```powershell
Set-Location 'C:\Users\vimal_z0\Desktop\loan-approval-bias-auditor'
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open http://localhost:8501 in a browser. Keep the terminal open. Press Ctrl+C in the terminal to stop. If port 8501 is occupied, append `--server.port 8502`, then visit http://localhost:8502. No activation, retraining, or downloads are required when the saved artifacts are present.

The app caches report files and loads models only for the scenario page. The first Python/Streamlit startup can be slower than later interactions; no instant-start benchmark is claimed.

## New machine setup

Python 3.10+ is the intended baseline; the delivered environment was tested on Python 3.13.7. `requirements-lock.txt` records the tested environment; platform-specific pins may not transfer across Python versions or operating systems.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Rebuild artifacts (only when intentionally rebuilding)

```powershell
.\.venv\Scripts\python.exe src/load_data.py
.\.venv\Scripts\python.exe src/train_baseline.py
.\.venv\Scripts\python.exe src/audit_bias.py
.\.venv\Scripts\python.exe src/finish_project.py
.\.venv\Scripts\python.exe src/verify_project.py
```

The final script overwrites the final reports with a reproducible evaluation of the fixed design. Do not repeatedly tune against test results. Steps 2–3 reserve the test set; Step 4's final evaluation explicitly supersedes that reservation. Existing Step 2/3 reports remain historical baseline artifacts; the frontend uses `reports/final/`.

## Project layout

```text
data/                         cached source, cleaned CSV, immutable split membership
notebooks/                    optional exploration
src/load_data.py              data loading and validation
src/train_baseline.py         preprocessing and baseline fitting
src/audit_bias.py             original validation fairness audit
src/finish_project.py         reweighting, final evaluation, SHAP caching
src/verify_project.py         artifact and frontend smoke checks
models/                       fitted pipelines and SHAP explainers
reports/fairness/             Step 3 baseline audit
reports/final/                final comparisons, predictions, explanations, manifest
.streamlit/config.toml        local server and theme settings
app.py                        five-page Streamlit frontend
requirements.txt              project dependencies
requirements-lock.txt         tested installed versions
PRESENTATION_GUIDE.md          walkthrough, terminology, and interview questions
```

## Method

- 1,000 records, target-stratified 600/200/200 split with random seed 42.
- 18 predictors: target and raw/derived protected columns are excluded. This alone does not ensure fairness.
- Training-only median/mode imputation and one-hot encoding handle numerical/categorical input. Trees need no scaling.
- XGBoost: 150 trees, depth 3, learning rate 0.05, row/column subsampling 0.8. Threshold fixed at 0.5.
- Majority-class dummy anchors the baseline: 70% accuracy but 50% balanced accuracy.
- Fairlearn MetricFrame calculates group counts, approval rate, accuracy, TPR, and FPR.
- Demographic parity gap = max group approval rate − min group approval rate.
- Equalized odds gap = max(TPR gap, FPR gap).
- Mitigation weights each training example by `P(age group) × P(label) / P(age group, label)`. This makes age group and label independent in the weighted training distribution. It does not force model predictions to satisfy a fairness constraint. Implemented directly in pandas/NumPy; no claim that this is a Fairlearn mitigation estimator.
- Only one fixed reweighted candidate is fitted; same settings as the baseline. All candidates and both splits are reported. No test-based hyperparameter selection.
- SHAP TreeExplainer uses raw log-odds and tree-path-dependent reference semantics. One-hot contributions are summed back to original features. Sigmoid(base + contributions) is checked against every saved score.
- Wilson intervals show approximate 95% uncertainty for individual group rates. They are not confidence intervals for disparities or significance tests.

## Measured results

| Final test metric (200 records) | Baseline | Reweighted |
|---|---:|---:|
| Accuracy | 74.5% | 74.5% |
| Balanced accuracy | 66.07% | 65.60% |
| ROC-AUC | 0.7760 | 0.7643 |
| Age demographic parity gap | 11.26 pp | 8.20 pp |
| Age equalized odds gap | 16.17 pp | 18.84 pp |
| False approvals (bad risk → approved) | 33 | 34 |
| False denials (good risk → denied) | 18 | 17 |

On validation, accuracy rose from 76.5% to 77.0%, but the age demographic parity gap increased from 9.91 to 11.79 pp and equalized odds increased from 8.65 to 10.67 pp. The intervention did not consistently reduce disparities. Present this as a measured trade-off, not a uniformly improved or certified-fair model. No production model is promoted.

## Frontend walkthrough

1. **Overview:** compare accuracy, balanced accuracy, ROC-AUC, fairness gaps, and confusion matrices.
2. **Fairness audit:** choose age or unverified gender codebook groups; inspect denominators and uncertainty whiskers.
3. **Explain a decision:** choose baseline/reweighted, inspect global SHAP, select a record, read a local waterfall in log-odds.
4. **Try a scenario:** alter amount and duration for a saved record; compare both scores. This does not alter reports or prove causal effects.
5. **Project guide:** review the method, training weights, provenance, and limitations.

## Limitations worth explaining

Small historical data, label/protected-attribute quality, arbitrary age bins, limited subgroup counts, possible proxy features, uncalibrated scores, and one split all limit conclusions. No causal discrimination test, intersectional audit, deployment validation, or legal compliance claim is made. SHAP is a model explanation, not a causal explanation. The dataset's asymmetric mistake costs are not optimized by the fixed threshold used here.

## Common errors

- Missing package: install requirements with the project's `.venv` Python.
- Missing artifacts: execute the rebuild commands in order.
- SHAP/XGBoost incompatibility on a different machine: use the tested versions in the lock file or rebuild and run `verify_project.py` after changing versions.
- Port in use: use `--server.port 8502`.
- Old reports after intentional rebuilding: refresh the browser; report caches use file modification times.
- Do not run `python app.py`; use `python -m streamlit run app.py`.
- A scenario score changed but saved metrics did not: expected; scenarios are separate from the frozen evaluation.
- Large table text wraps in a terminal: open CSVs in VS Code or use the dashboard.

## Sources and attribution

- [UCI Statlog German Credit dataset](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), Hans Hofmann, UCI Machine Learning Repository; DOI 10.24432/C5NC77.
- [UCI corrected coding discussion](https://archive.ics.uci.edu/dataset/573/south+german+credit+update).
- [Fairlearn equalized odds definition](https://fairlearn.org/v0.13/api_reference/generated/fairlearn.metrics.equalized_odds_difference.html).
- [Reweighing concept](https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.Reweighing.html).
- [SHAP TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html).
- [Streamlit caching](https://docs.streamlit.io/develop/concepts/architecture/caching).

The dataset is distributed under UCI's CC BY 4.0 terms; retain attribution and check the linked source for the applicable dataset license. Code and data should not be represented as an original collected lending dataset.

