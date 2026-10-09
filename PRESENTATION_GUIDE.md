# Explain this project confidently

## Your 60-second opening

"I built a reproducible audit workflow around a loan-approval proxy. I trained an XGBoost credit-risk model, used Fairlearn to compare approval and error rates across groups, tried age-based sample reweighting, and used SHAP to explain individual scores. The Streamlit dashboard loads saved models and reports, so the demo doesn't retrain anything. On 200 held-out test records, accuracy stayed at 74.5%. Age approval-rate disparity fell from 11.26 to 8.20 percentage points, but equalized-odds disparity increased. The main lesson is that fairness is a choice of criteria and a measured trade-off, not a single score."

Immediately add: "This is a demonstration: the labels are credit risk, not actual approvals, and the gender codebook is unreliable."

## Five-minute live demo

1. **0:00–0:45, Overview:** say what the label means, point to the held-out test split, and explain that 70% is the trivial accuracy reference.
2. **0:45–1:45, Fairness audit / age:** compare approval rates, then TPR and FPR. Point to group counts and uncertainty bars. Explain the conflicting mitigation result.
3. **1:45–2:45, Explain a decision:** choose a model and record. Positive SHAP values push the score toward good risk, negative values toward bad risk. The base plus contributions equals the raw margin; sigmoid converts it to the score.
4. **2:45–3:30, Try a scenario:** change duration or amount, submit, and compare model scores. Say this is sensitivity to a hypothetical input, not a causal promise.
5. **3:30–4:15, Project guide:** show the sample-weight formula and say it uses training data only. Age is not required at prediction time.
6. **4:15–5:00, Limitations:** discuss the gender-codebook issue, small groups, proxies, and the absence of an unambiguous fairness improvement.

## Every step in plain language

| Step | What you did | Why |
|---|---|---|
| 1 | Cleaned names, defined the proxy label and group columns | Make assumptions explicit and reproducible |
| 2 | Split 600/200/200, fitted preprocessing + XGBoost | Avoid leaking held-out information into training |
| 3 | Used MetricFrame for group metrics | Aggregate accuracy can hide uneven outcomes |
| 4 | Fitted a second model with age/label sample weights | Try changing the influence of training examples |
| 5 | Calculated and cached SHAP for both models | Explain global patterns and individual scores |
| 6 | Built the Streamlit frontend | Let an audience inspect results interactively |
| 7 | Checked artifacts, predictions, explanations, and UI controls | Ensure the saved results agree and the demo works |

## Terms you must be able to define

- **False positive:** bad-risk record predicted approved. Here approval is positive.
- **False negative:** good-risk record predicted denied.
- **TPR / recall:** TP / (TP + FN), approval rate among good-risk records.
- **FPR:** FP / (FP + TN), approval rate among bad-risk records.
- **Precision:** TP / (TP + FP), good-risk fraction among predicted approvals.
- **Balanced accuracy:** average of good-risk recall and bad-risk recall; reduces majority-class dominance.
- **ROC-AUC:** how well scores rank positives above negatives across thresholds; not fairness and not calibration.
- **Demographic parity:** matching overall approval rates across groups, regardless of label.
- **Equalized odds:** matching both TPR and FPR across groups, conditional on the recorded label.
- **Percentage points:** 11.26% minus 8.20% = 3.06 percentage points, not 3.06 percent relative reduction.
- **Reweighting:** weights change each example's contribution to training loss; no examples are added or relabeled.
- **SHAP:** additive feature attributions for a fitted model. Our values use log-odds, not probability percentage points.
- **Calibration:** among scores near 0.8, approximately 80% should be positive. We did not establish this property.
- **Proxy feature:** a feature correlated with a protected attribute; removing the attribute itself may not remove its information.

## Reweighting example

Suppose 20% of training records belong to an age group and 70% have a good-risk label. Independence would put 14% in the joint cell. If only 10% are observed there, each record in that cell receives weight 14/10 = 1.4. Overrepresented cells receive weights below 1. Our code does this for all three age groups and both labels, using the actual training counts.

This equalizes weighted label prevalence by age in training. It does NOT guarantee equal prediction rates, and it does NOT directly optimize equalized odds.

## Questions a judge might ask

**Why XGBoost?** It handles nonlinear tabular patterns and interactions, supports sample weights, and works with Tree SHAP. I used shallow trees to keep the small-data baseline modest, not a large tuning search.

**Why no scaling?** Tree splits do not require standardized numeric features. Categorical strings are still one-hot encoded.

**Why three splits?** Train fits model/weights, validation supports development, and the test set is reserved until the method and threshold are fixed. After reporting test results, it must not become a tuning set.

**Did mitigation succeed?** Not uniformly. Test age approval disparity decreased by 3.06 pp at unchanged accuracy, but age equalized-odds disparity increased by 2.67 pp; both age gaps worsened on validation. I report the conflict rather than selecting only favorable metrics.

**Why not optimize gender fairness?** The combined source sex/status codes cannot reliably identify gender. Optimizing an unreliable label would produce misleading claims. I keep that view as a labeled tooling demonstration.

**Is excluding age enough?** No. Employment duration, housing, or financial history may correlate with age. The audit checks outputs despite exclusion of the direct attribute.

**Can you call it fair?** No. Fairness criteria reflect different goals, results are uncertain, and real lending needs context and suitable data. The dashboard is an audit tool, not a fairness certification.

**Does SHAP show what causes approval?** No. It attributes a model output under an explanation convention; it does not identify real-world causal effects or prove that a feature is justified.

**Why keep the reweighted model if it isn't better?** It makes the experiment reproducible and shows the trade-offs. The project does not promote it for deployment.

**What would you improve next?** Obtain suitable approval/outcome data with reliable group labels; use larger groups and repeated evaluation; test intersectional disparities, calibration, alternative mitigation methods, and decision costs. Choose fairness goals with stakeholders before optimizing them.

## Honest resume bullet

"Built an XGBoost credit-risk bias-auditing dashboard with Fairlearn, age-based sample reweighting, and SHAP explanations; evaluated performance and fairness trade-offs on a held-out test set and cached artifacts for interactive Streamlit exploration."

Avoid: "eliminated bias," "proved gender discrimination," "production-ready loan approval," or "achieved fairness without trade-offs."

## Last-minute rehearsal (15 minutes)

1. Run the dashboard and visit all five pages.
2. Explain FP and FN without looking at notes.
3. Explain the two fairness definitions and the reweighting formula.
4. Explain one SHAP waterfall, including its units.
5. Say the dataset limitations aloud.
6. Practice the opening once in under 60 seconds.

If the browser fails during the event, open `reports/final/performance.csv`, `fairness.csv`, and this guide. All results are saved offline.
