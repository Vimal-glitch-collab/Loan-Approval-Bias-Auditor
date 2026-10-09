"""Step 1: prepare UCI German Credit for a demonstration audit."""
from pathlib import Path
from urllib.request import urlopen
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/144/statlog+german+credit+data.zip"
COLUMNS = [
    "checking_account_status", "loan_duration_months", "credit_history",
    "loan_purpose", "credit_amount", "savings_status", "employment_duration",
    "installment_rate", "personal_status_sex", "other_debtors",
    "residence_duration", "property_category", "age", "other_installment_plans",
    "housing", "existing_credit_count", "job_category", "dependent_count",
    "telephone", "foreign_worker", "credit_risk_code",
]
TARGET = "loan_approved"
# This is the ORIGINAL codebook mapping, not verified gender information.
GENDER_MAP = {"A91": "male", "A92": "female", "A93": "male",
              "A94": "male", "A95": "female"}
PROTECTED = ["gender_codebook_proxy", "age_group"]


def main():
    import io
    import zipfile

    for folder in ["data", "notebooks", "src", "models", "reports"]:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)

    # Cache original text so subsequent runs work offline.
    raw_path = ROOT / "data/german.data"
    if not raw_path.exists():
        with urlopen(URL, timeout=60) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            raw_path.write_bytes(archive.read("german.data"))

    df = pd.read_csv(raw_path, sep=r"\s+", header=None,
                     na_values=["?"], keep_default_na=True)
    if df.shape != (1000, 21):
        raise ValueError(f"Unexpected source shape: {df.shape}")
    df.columns = COLUMNS
    print("Raw shape:", df.shape)

    for column in df.columns:
        if pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].str.strip()
    if df.isna().any().any():
        raise ValueError("Unexpected missing data; inspect before imputing.")
    if set(df["credit_risk_code"].unique()) != {1, 2}:
        raise ValueError("Unexpected target codes.")
    if not df["age"].between(18, 100).all():
        raise ValueError("Unexpected ages; inspect the source.")

    # Demonstration convention: good credit risk -> approved proxy.
    # These labels are NOT historical loan approval decisions.
    df[TARGET] = df["credit_risk_code"].map({1: 1, 2: 0}).astype("int64")
    df = df.drop(columns="credit_risk_code")  # Prevent duplicate-target leakage.

    df["gender_codebook_proxy"] = df["personal_status_sex"].map(GENDER_MAP)
    # Non-overlapping intervals: <25, 25 <= age <40, age >=40.
    df["age_group"] = pd.cut(df["age"], bins=[0, 25, 40, float("inf")],
                              labels=["<25", "25-39", "40+"], right=False)
    if df[PROTECTED].isna().any().any():
        raise ValueError("Unmapped protected attribute values.")

    # Keep attributes for auditing, but exclude them from baseline predictors.
    excluded = [TARGET, "gender_codebook_proxy", "age_group",
                "age", "personal_status_sex"]
    features = [column for column in df.columns if column not in excluded]

    schema = pd.DataFrame({"dtype": df.dtypes.astype(str),
                           "missing_count": df.isna().sum()})
    print("\nCleaned shape:", df.shape)
    print("\nDtypes and missing values:\n", schema.to_string())
    print("\nTarget counts:\n", df[TARGET].value_counts().sort_index())
    for column in PROTECTED:
        print(f"\n{column} counts:\n{df[column].value_counts().sort_index()}")
    print("\nDuplicate rows (retained):", df.duplicated().sum())

    # No scaling or encoding yet: fit learned transformations after splitting.
    output = ROOT / "data/german_credit_clean.csv"
    df.to_csv(output, index=False)
    schema.to_csv(ROOT / "reports/data_quality.csv", index_label="column")
    metadata = {
        "source": URL, "target": TARGET,
        "target_meaning": "1=good credit risk; 0=bad credit risk; approval proxy only",
        "protected_attributes": PROTECTED, "feature_columns": features,
        "age_groups": "<25; 25 <= age <40; age >=40",
        "limitations": [
            "No observed loan approval labels.",
            "Original sex/status coding is disputed. gender_codebook_proxy is unverified.",
            "Gender proxy comparisons cannot establish real gender disparities.",
            "CSV does not preserve pandas categorical dtypes.",
        ],
        "coding_correction_source": "https://archive.ics.uci.edu/dataset/573/south+german+credit+update",
    }
    (ROOT / "reports/data_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
