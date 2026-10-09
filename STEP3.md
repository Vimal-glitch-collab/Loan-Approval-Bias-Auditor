# Step 3: Fairlearn audit

Budget: 60 minutes, including interpreting metrics. This step audits saved validation predictions without retraining or inspecting the test set. Fairlearn MetricFrame groups the same metrics by age group and the unverified gender codebook proxy. Tables and charts are saved for later dashboard use.

## Run

The complete commented implementation is in `src/audit_bias.py`.

```powershell
Set-Location 'C:\Users\vimal_z0\Desktop\loan-approval-bias-auditor'
.\.venv\Scripts\python.exe src/audit_bias.py
```

## Read the metrics

- Selection rate: fraction predicted approved (a credit-risk proxy).
- True-positive rate: fraction of good-risk records predicted approved; denominator is positive_count.
- False-positive rate: fraction of bad-risk records predicted approved; denominator is negative_count.
- Demographic parity difference: maximum minus minimum group selection rate.
- Equalized odds difference: larger of the true-positive-rate gap and false-positive-rate gap.
- Differences range from 0 to 1; multiply by 100 for percentage points. Zero means equal observed rates under that metric, not proof of fairness.

Source: https://fairlearn.org/v0.13/api_reference/generated/fairlearn.metrics.equalized_odds_difference.html

## Outputs

Open `reports/fairness/` for group_metrics.csv, fairness_summary.csv, overall_metrics.csv, individual attribute metric tables, two self-contained interactive HTML charts, baseline_group_rates.png, and audit_metadata.json.

No confidence intervals or statistical significance tests are included. The small_denominator flag marks fewer than 20 positives or negatives as a descriptive caution, not a formal threshold. Undefined conditional rates remain missing. Separate attribute audits can hide intersectional differences. Demographic parity and equalized odds represent different objectives and can conflict. The gender codebook is disputed and the results cannot establish real gender disparities. The target is not an observed approval decision.

## Common errors

- Missing prediction/split files: run `src/train_baseline.py` first.
- Missing Fairlearn, Plotly, or Matplotlib: run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- Alignment check fails: regenerate the Step 2 artifacts together; do not combine predictions and splits from different runs.
- A rate is blank: that group has no records for the required denominator; do not replace it with zero.
- HTML shows source code in VS Code: open the file with a web browser; Plotly is embedded for offline use.

Stop here before mitigation (Step 4).
