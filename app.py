"""Applicant-facing demo. All 18 model inputs are supplied by the form."""
from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Loan assessment", page_icon="◈", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
[data-testid="stSidebar"], [data-testid="stToolbar"] {display:none;}
.block-container {max-width:1120px;padding-top:2.5rem;padding-bottom:3rem;}
h1 {font-size:2.8rem !important;letter-spacing:-1.5px;font-weight:700 !important;}
h3 {font-size:1.15rem !important;letter-spacing:-.3px;}
.brand {font-size:12px;letter-spacing:2px;font-weight:700;color:#26796b;margin-bottom:20px;}
.intro {font-size:17px;color:#65716c;line-height:1.6;max-width:650px;margin-bottom:24px;}
[data-testid="stForm"] {background:#fff;border:1px solid #e0e6df;border-radius:16px;padding:26px;}
.result {background:#f0f6f1;border:1px solid #d2e3d7;border-radius:16px;padding:26px;margin-top:10px;}
.result h2 {font-size:25px;letter-spacing:-.6px;margin-top:8px;}
.result p {color:#54635b;line-height:1.6;}
.note {font-size:12px;color:#69756f;line-height:1.6;margin-top:24px;}
div[data-testid="stFormSubmitButton"] button {border-radius:8px;min-height:46px;font-weight:600;}
</style>""", unsafe_allow_html=True)

# Values follow the original UCI codebook used to train this saved model.
# Source: https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data
CHOICES = {
    "loan_purpose": {"Car — new":"A40", "Car — used":"A41", "Furniture or equipment":"A42", "TV or radio":"A43", "Household appliances":"A44", "Repairs":"A45", "Education":"A46", "Retraining":"A48", "Business":"A49", "Other":"A410"},
    "checking_account_status": {"No checking account":"A14", "Balance below 0":"A11", "Balance from 0 to under 200":"A12", "Balance 200 or more / established salary assignment":"A13"},
    "savings_status": {"Under 100":"A61", "100 to under 500":"A62", "500 to under 1,000":"A63", "1,000 or more":"A64", "Unknown / no savings account":"A65"},
    "employment_duration": {"Unemployed":"A71", "Less than 1 year":"A72", "1 to under 4 years":"A73", "4 to under 7 years":"A74", "7 years or more":"A75"},
    "credit_history": {"Existing credit paid on time":"A32", "No previous credit / all repaid":"A30", "All credit at this bank repaid":"A31", "Past payment delays":"A33", "Critical account / credit elsewhere":"A34"},
    "other_debtors": {"None":"A101", "Co-applicant":"A102", "Guarantor":"A103"},
    "property_category": {"Real estate":"A121", "Savings agreement / life insurance":"A122", "Car or other property":"A123", "Unknown / no property":"A124"},
    "other_installment_plans": {"None":"A143", "Bank":"A141", "Stores":"A142"},
    "housing": {"Rent":"A151", "Own":"A152", "Provided at no cost":"A153"},
    "job_category": {"Skilled employee":"A173", "Management / self-employed / highly qualified":"A174", "Unskilled resident":"A172", "Unemployed / unskilled non-resident":"A171"},
    "telephone": {"Yes":"A192", "No":"A191"},
    "foreign_worker": {"Yes":"A201", "No":"A202"},
}


def choice(label, column, help=None):
    selected = st.selectbox(label, list(CHOICES[column]), key=column, help=help)
    return CHOICES[column][selected]


@st.cache_resource
def load_model(modified):
    # Baseline retained: reweighting did not consistently improve fairness.
    return joblib.load(ROOT / "models/baseline_pipeline.joblib")


st.markdown('<div class="brand">LOAN ASSESSMENT / DEMO</div>', unsafe_allow_html=True)
st.title("A clearer look at your loan request.")
st.markdown('<p class="intro">Enter the details below to get a credit-risk estimate. Start with the loan, then add your financial circumstances.</p>', unsafe_allow_html=True)
st.caption("Use example figures in historical dataset units, not INR or today's currency values. No name or contact details are needed.")

left, right = st.columns([1.7, 1], gap="large")
values = {}
with left:
    with st.form("application", clear_on_submit=False):
        st.subheader("01 / Your loan")
        a, b = st.columns(2)
        with a:
            values["credit_amount"] = st.number_input("Requested amount", min_value=250, max_value=18500, value=3000, step=100, help="Example amount in the historical dataset's units.")
            values["loan_purpose"] = choice("Purpose", "loan_purpose")
        with b:
            values["loan_duration_months"] = st.number_input("Repayment period (months)", min_value=4, max_value=72, value=24, step=1)
            values["installment_rate"] = st.selectbox("Repayment burden", [1, 2, 3, 4], index=1, format_func=lambda x: {1:"Low — band 1", 2:"Moderate — band 2", 3:"High — band 3", 4:"Highest — band 4"}[x], help="Choose a relative installment-burden band used by this historical dataset; this is not a calculated monthly payment.")
        st.divider()
        st.subheader("02 / Your finances")
        a, b = st.columns(2)
        with a:
            values["checking_account_status"] = choice("Checking account balance", "checking_account_status")
            values["credit_history"] = choice("Credit history", "credit_history")
            values["existing_credit_count"] = st.number_input("Existing credits at this bank", min_value=1, max_value=4, value=1, help="Historical dataset categories range from 1 to 4.")
        with b:
            values["savings_status"] = choice("Savings balance", "savings_status")
            values["other_installment_plans"] = choice("Other installment plans", "other_installment_plans")
            values["other_debtors"] = choice("Additional borrower or guarantor", "other_debtors")
        st.divider()
        st.subheader("03 / Work & household")
        a, b = st.columns(2)
        with a:
            values["job_category"] = choice("Employment type", "job_category")
            values["housing"] = choice("Housing", "housing")
            values["property_category"] = choice("Most valuable property", "property_category")
            values["telephone"] = choice("Telephone registered in your name", "telephone")
        with b:
            values["employment_duration"] = choice("Time with current employer", "employment_duration")
            values["residence_duration"] = st.selectbox("Time at current residence", [1, 2, 3, 4], index=1, format_func=lambda x: {1:"Shortest duration — band 1", 2:"Shorter duration — band 2", 3:"Longer duration — band 3", 4:"Longest duration — band 4"}[x], help="The source uses four ordered residence-duration bands, not exact years.")
            values["dependent_count"] = st.selectbox("People financially supported", [1, 2], format_func=lambda x: "1" if x == 1 else "2 or more")
            values["foreign_worker"] = choice("Foreign worker status", "foreign_worker", help="A field in the historical dataset. This demo should not be used for real lending decisions.")
        st.write("")
        submitted = st.form_submit_button("View my assessment →", type="primary", width="stretch")

with right:
    st.subheader("Your assessment")
    if not submitted:
        st.markdown('<div class="result"><div class="brand">READY WHEN YOU ARE</div><h2>Let’s start with your details.</h2><p>Complete the form and select <strong>View my assessment</strong>. Your result will appear here.</p></div>', unsafe_allow_html=True)
    else:
        path = ROOT / "models/baseline_pipeline.joblib"
        if not path.exists():
            st.error("The assessment service is unavailable. Please ask the project owner to restore the saved model.")
        else:
            metadata = json.loads((ROOT / "models/baseline_metadata.json").read_text())
            row = pd.DataFrame([values])[metadata["feature_columns"]]
            model = load_model(path.stat().st_mtime_ns)
            score = float(model.predict_proba(row)[0, 1])
            positive = score >= metadata["threshold"]
            title = "Favourable estimate" if positive else "Unfavourable estimate"
            text = "These details fall on the lower credit-risk side of this demonstration's assessment." if positive else "These details fall on the higher credit-risk side of this demonstration's assessment."
            st.markdown(f'<div class="result"><div class="brand">ASSESSMENT COMPLETE</div><h2>{title}</h2><p>{text}</p><p><strong>{values["credit_amount"]:,}</strong> requested over <strong>{values["loan_duration_months"]} months</strong>.</p></div>', unsafe_allow_html=True)
            st.caption("This result reflects your last submission. Edit the form and submit again to update it.")
    st.markdown('<p class="note">Educational demonstration using historical credit-risk data. This is not a loan approval, eligibility guarantee, or financial advice. Form submissions are not written to project files.</p>', unsafe_allow_html=True)
