# Step 2: baseline model

Budget: 75 minutes including reviewing the code and interpreting results.

This step uses the cleaned German Credit dataset, splits it into 60% training, 20% validation, and 20% test, fits preprocessing on training records only, and trains XGBoost plus a majority-class reference model. The fitted pipeline and metric tables are saved for reuse. The test set stays unevaluated until the final comparison.

## Run in VS Code PowerShell

```powershell
Set-Location 'C:\Users\vimal_z0\Desktop\loan-approval-bias-auditor'
.\.venv\Scripts\python.exe src/train_baseline.py
```

The complete, commented code is in `src/train_baseline.py`. Its `ROOT` is derived from the script location, so moving the whole project does not require hardcoded Python path changes.

## Expected output and saved files

Expect 600 training records, 200 validation records, 200 reserved test records, and 18 input predictors. The majority dummy has 70% accuracy, 50% balanced accuracy, and 0.50 ROC-AUC. XGBoost scores are printed by the script.

- `models/baseline_pipeline.joblib`: fitted preprocessing and XGBoost.
- `models/majority_dummy.joblib`: fitted comparison model.
- `models/baseline_metadata.json`: features, versions, threshold, dataset hash, and limitations.
- `reports/baseline_metrics.csv`: training and validation metrics.
- `reports/baseline_train_predictions.csv` and `reports/baseline_validation_predictions.csv`: aligned labels, scores, predictions, and audit attributes.
- `data/splits/train.csv`, `validation.csv`, `test.csv`: reproducible splits with row IDs.

Accuracy measures overall correctness; balanced accuracy averages recall for each class. Recall here measures the fraction of good-risk records predicted approved. Precision measures the good-risk fraction among predicted approvals. Compare training and validation results to identify possible overfitting. The 0.5 threshold is a demonstration default, not an optimized lending policy.

## Common errors

- Missing CSV: run `.\.venv\Scripts\python.exe src/load_data.py` first.
- Missing package: run `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- `sparse_output` not recognized: update scikit-learn in this environment.
- Missing `.venv` interpreter: confirm the terminal is in the Desktop project folder and the environment exists.

## Limitations

The target is a good-credit-risk proxy, not an observed approval decision. Gender is derived from a disputed codebook and cannot support real gender-disparity claims. Age, age group, gender proxy, and raw sex/status are excluded from predictors, but this does not guarantee fairness. No fairness audit, mitigation, SHAP, or dashboard is implemented in this step.
