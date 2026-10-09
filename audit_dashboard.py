"""Streamlit frontend. Run: python -m streamlit run app.py"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports/final"
st.set_page_config(page_title="Loan Approval Bias Auditor", page_icon="⚖️", layout="wide")
st.markdown("""<style>
.block-container {max-width:1380px;padding-top:2rem;padding-bottom:3rem;}
h1 {letter-spacing:-1.4px;font-weight:750 !important;}
h2,h3 {letter-spacing:-.5px;}
[data-testid="stMetric"] {background:#f0f5f5;border:1px solid #dce8e7;border-radius:12px;padding:16px;}
[data-testid="stSidebar"] {background:#eef4f3;}
.eyebrow {color:#167b72;font-size:12px;font-weight:750;letter-spacing:2px;margin-bottom:8px;}
.subtitle {color:#52646a;font-size:17px;max-width:900px;line-height:1.6;}
</style>""", unsafe_allow_html=True)

if not (REPORTS / "manifest.json").exists() or not (REPORTS / "shap_local.csv").exists():
    st.title("Loan Approval Bias Auditor")
    st.info("Build the saved reports first. Run this command in the project terminal:")
    st.code(r".\.venv\Scripts\python.exe src/finish_project.py", language="powershell")
    st.stop()


@st.cache_data
def csv_file(name, modified):
    return pd.read_csv(REPORTS / name)


def read(name):
    return csv_file(name, (REPORTS / name).stat().st_mtime_ns)


@st.cache_resource
def pipeline(name, modified):
    filename = "baseline_pipeline.joblib" if name == "Baseline" else "reweighted_pipeline.joblib"
    return joblib.load(ROOT / "models" / filename)


def get_pipeline(name):
    filename = "baseline_pipeline.joblib" if name == "Baseline" else "reweighted_pipeline.joblib"
    return pipeline(name, (ROOT / "models" / filename).stat().st_mtime_ns)


@st.cache_resource
def explainer(name, modified):
    return joblib.load(ROOT / f"models/{name}_explainer.joblib")


def pretty(name):
    return name.replace("_", " ").capitalize()


COLORS = {"Baseline": "#375b8c", "Reweighted": "#16877a"}
manifest = json.loads((REPORTS / "manifest.json").read_text())
features = manifest["features"]
with st.sidebar:
    st.markdown("### ⚖ Loan audit lab")
    st.caption("GERMAN CREDIT · DEMONSTRATION")
    page = st.radio("Workspace", ["Overview", "Fairness audit", "Explain a decision", "Try a scenario", "Project guide"])
    split = st.selectbox("Evaluation sample", ["test", "validation"], format_func=lambda v: "Final test · 200 records" if v == "test" else "Validation · 200 records")
    st.divider()
    st.caption("600 training · 200 validation · 200 test")
    st.caption("XGBoost · Fairlearn · SHAP")
    st.caption("Saved artifacts • No training on page load")
    st.warning("Credit-risk labels stand in for approvals. This is an educational audit, not a lending system.")

st.markdown('<div class="eyebrow">RESPONSIBLE ML / MODEL AUDIT</div>', unsafe_allow_html=True)
st.title("Loan Approval Bias Auditor")
st.markdown('<div class="subtitle">Compare predictive performance, inspect group disparities, and understand the features behind each score.</div>', unsafe_allow_html=True)
st.caption(f"{split.title()} results · fixed decision threshold 0.50 · age-based training reweighting")
performance = read("performance.csv").query("split == @split")
fairness = read("fairness.csv").query("split == @split")
groups = read("group_metrics.csv").query("split == @split")


def comparison_chart(table, y, title, percent=True):
    fig = px.bar(table, x="group", y=y, color="model", barmode="group",
                 color_discrete_map=COLORS, title=title,
                 category_orders={"group": ["<25", "25-39", "40+", "female", "male"]},
                 hover_data=["count", "positive_count", "negative_count"],
                 error_y=table[y+"_high"]-table[y] if y+"_high" in table else None,
                 error_y_minus=table[y]-table[y+"_low"] if y+"_low" in table else None)
    fig.update_layout(template="plotly_white", legend_title_text="", xaxis_title="", yaxis_title="Rate")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return fig


if page == "Overview":
    st.subheader("Performance with the trade-offs visible")
    baseline = performance.set_index("model").loc["Baseline"]
    weighted = performance.set_index("model").loc["Reweighted"]
    age = fairness.query("attribute == 'age_group'").set_index("model")
    cols = st.columns(4)
    cols[0].metric("Baseline accuracy", f"{baseline.accuracy:.1%}")
    cols[1].metric("Reweighted accuracy", f"{weighted.accuracy:.1%}", f"{(weighted.accuracy-baseline.accuracy)*100:+.1f} pp")
    cols[2].metric("Age approval-rate gap", f"{age.loc['Reweighted', 'demographic_parity_difference']*100:.1f} pp", f"{(age.loc['Reweighted', 'demographic_parity_difference']-age.loc['Baseline', 'demographic_parity_difference'])*100:+.1f} pp", delta_color="inverse")
    cols[3].metric("Age equalized-odds gap", f"{age.loc['Reweighted', 'equalized_odds_difference']*100:.1f} pp", f"{(age.loc['Reweighted', 'equalized_odds_difference']-age.loc['Baseline', 'equalized_odds_difference'])*100:+.1f} pp", delta_color="inverse")
    st.caption("Fairness cards show the reweighted model; changes are relative to baseline. pp = percentage points. Smaller gaps are closer to parity, not proof of fairness.")
    left, right = st.columns([1.2, 1])
    with left:
        melted = performance.melt(id_vars=["model"], value_vars=["accuracy", "balanced_accuracy", "roc_auc"], var_name="metric", value_name="score")
        fig = px.bar(melted, x="metric", y="score", color="model", barmode="group", color_discrete_map=COLORS, title="Beyond headline accuracy")
        fig.update_layout(template="plotly_white", xaxis_title="", legend_title_text="")
        fig.update_yaxes(range=[0,1])
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("#### What changed?")
        st.write("Both models use the same 18 predictors and XGBoost settings. The second model changes only the influence of training records, using age × label weights.")
        st.write("Age and the gender proxy are excluded from prediction inputs. Other features can still carry related information.")
        st.info("The majority-class reference scores 70% accuracy. Balanced accuracy and ROC-AUC help distinguish useful prediction from simply approving everyone.")
    st.markdown("#### Errors have different consequences")
    for col, name in zip(st.columns(2), ["Baseline", "Reweighted"]):
        row = performance.set_index("model").loc[name]
        matrix = [[row.tn, row.fp], [row.fn, row.tp]]
        with col:
            fig = px.imshow(matrix, text_auto=True, x=["Predicted deny", "Predicted approve"], y=["Actual bad risk", "Actual good risk"], color_continuous_scale="Teal", title=name)
            fig.update_layout(coloraxis_showscale=False, height=310)
            st.plotly_chart(fig, width="stretch")
    st.download_button("Download performance table", performance.to_csv(index=False), "performance.csv", "text/csv")

elif page == "Fairness audit":
    st.subheader("Who receives different outcomes?")
    attribute = st.selectbox("Compare groups", ["age_group", "gender_codebook_proxy"], format_func=lambda x: "Age groups" if x == "age_group" else "Unverified gender codebook groups (demo only)")
    if attribute.startswith("gender"):
        st.warning("The source sex/marital-status coding is disputed. These groups demonstrate the calculation and cannot establish actual gender disparities. Mitigation targets age only.")
    table = groups.query("attribute == @attribute")
    gap = fairness.query("attribute == @attribute")
    st.dataframe(gap[["model", "demographic_parity_difference", "equalized_odds_difference", "true_positive_rate_difference", "false_positive_rate_difference"]].set_index("model").style.format("{:.3f}"), width="stretch")
    st.caption("Gaps are fractions; 0.10 equals 10 percentage points. Demographic parity compares approval rates. Equalized odds uses the larger of the TPR and FPR gaps.")
    st.plotly_chart(comparison_chart(table, "selection_rate", "Predicted approval proxy rate"), width="stretch")
    for col, metric, title in zip(st.columns(2), ["true_positive_rate", "false_positive_rate"], ["Good-risk records approved (TPR)", "Bad-risk records approved (FPR)"]):
        with col:
            st.plotly_chart(comparison_chart(table, metric, title), width="stretch")
    st.caption("Whiskers: approximate 95% Wilson intervals for individual group rates. They are not confidence intervals for gaps or tests of discrimination.")
    if table.small_denominator.any():
        st.info("Some groups have fewer than 20 good-risk or bad-risk examples. One changed prediction can move their rates substantially.")
    st.dataframe(table[["model", "group", "count", "positive_count", "negative_count", "accuracy", "small_denominator"]], hide_index=True, width="stretch")
    st.download_button("Download group audit", table.to_csv(index=False), "group_audit.csv", "text/csv")

elif page == "Explain a decision":
    st.subheader("From a score to its feature contributions")
    model_name = st.selectbox("Model to explain", ["Baseline", "Reweighted"])
    global_shap = read("shap_global.csv").query("model == @model_name and split == @split").sort_values("mean_abs_shap").tail(12)
    fig = px.bar(global_shap, x="mean_abs_shap", y="feature", orientation="h", title="Global importance · mean absolute SHAP", color_discrete_sequence=[COLORS[model_name]])
    fig.update_layout(template="plotly_white", xaxis_title="Mean |contribution| in log-odds", yaxis_title="", height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption("One-hot category contributions are summed back to each original feature. Importance is computed on the selected sample; it is not a causal ranking.")
    cases = read("cases.csv").query("split == @split").set_index("row_id")
    row_id = st.selectbox("Record to explain", cases.index.tolist())
    local = read("shap_local.csv").query("model == @model_name and split == @split and row_id == @row_id").iloc[0]
    pred = read("predictions.csv").query("model == @model_name and split == @split and row_id == @row_id").iloc[0]
    cols = st.columns(3)
    cols[0].metric("Good-risk model score", f"{pred.score:.1%}")
    cols[1].metric("Predicted proxy decision", "Approve" if pred.y_pred else "Deny")
    cols[2].metric("Observed credit-risk label", "Good risk" if pred.y_true else "Bad risk")
    ordered = sorted(features, key=lambda f: abs(local[f]), reverse=True)
    top, remaining = ordered[:8], ordered[8:]
    contributions = [float(local[f]) for f in top] + [float(sum(local[f] for f in remaining))]
    fig = go.Figure(go.Waterfall(orientation="h", y=["Base value"]+[pretty(f) for f in top]+["Other features", "Final margin"],
        x=[float(local.base_value)]+contributions+[0], measure=["absolute"]+["relative"]*9+["total"],
        increasing={"marker":{"color":"#16877a"}}, decreasing={"marker":{"color":"#db775d"}}, totals={"marker":{"color":"#375b8c"}}))
    fig.update_layout(title="Why this model produced this score", xaxis_title="Raw margin (log-odds)", template="plotly_white", height=480, yaxis_autorange="reversed")
    st.plotly_chart(fig, width="stretch")
    st.caption("Positive contributions push toward good risk; negative contributions push toward bad risk. Sigmoid(base + all contributions) reconstructs the score. Scores are not calibrated probabilities.")
    with st.expander("Inspect the record and all contributions"):
        display = pd.DataFrame({"feature": features, "input": [str(cases.loc[row_id, f]) for f in features], "SHAP (log-odds)": [local[f] for f in features]})
        st.dataframe(display, hide_index=True, width="stretch")

elif page == "Try a scenario":
    st.subheader("Explore a hypothetical change")
    st.write("Start with a saved record and change the amount or duration. This is a model sensitivity demonstration, not financial advice or a causal prediction.")
    cases = read("cases.csv").query("split == @split").set_index("row_id")
    chosen = st.selectbox("Starting record", cases.index.tolist(), key="scenario_record")
    row = cases.loc[[chosen], features].copy()
    with st.form("scenario"):
        col1, col2 = st.columns(2)
        amount = col1.number_input("Credit amount (historical dataset units)", min_value=1, max_value=100000, value=int(row.credit_amount.iloc[0]))
        duration = col2.number_input("Duration (months)", min_value=1, max_value=120, value=int(row.loan_duration_months.iloc[0]))
        submitted = st.form_submit_button("Compare model scores", type="primary")
    if submitted:
        row["credit_amount"], row["loan_duration_months"] = amount, duration
        for col, name in zip(st.columns(2), ["Baseline", "Reweighted"]):
            model = get_pipeline(name)
            score = float(model.predict_proba(row)[0, 1])
            with col:
                st.metric(name, f"{score:.1%}")
                st.write("Predicted approval proxy" if score >= .5 else "Predicted denial proxy")
        st.caption("The other 16 input features stay fixed. Unusual combinations or values outside training coverage can produce unreliable scores. Saved reports and test results are unchanged.")
    with st.expander("View the starting features"):
        st.dataframe(cases.loc[[chosen], features], width="stretch")

else:
    st.subheader("The story behind the project")
    st.markdown("""
**Question:** How do predictive performance and group disparities change when we alter training sample weights?

1. **Data:** 1,000 German Credit records; good credit risk is used as an approval proxy.
2. **Split:** 600 train / 200 validation / 200 test, stratified by label with seed 42.
3. **Baseline:** training-only imputation and one-hot encoding, followed by XGBoost.
4. **Audit:** Fairlearn compares age groups and an unverified gender codebook grouping.
5. **Mitigation:** age × label independence weights are learned only on training records. Same architecture, new sample weights.
6. **Evaluation:** one fixed candidate and threshold 0.5; both models reported on the final test set without test-based tuning.
7. **Explanation:** Tree SHAP contributions for each model, verified against its predictions.

**Weight formula:** `P(age group) × P(label) / P(age group, label)`.
This balances the age/label relationship in the weighted training distribution. It does not guarantee equalized prediction rates on new data.

**Demographic parity** asks whether approval rates match. **Equalized odds** asks whether TPR and FPR match conditional on the recorded label. These criteria capture different concerns and can conflict.
""")
    st.warning("The dataset is historical and small. Its target is not observed approval, and its combined sex/status coding is disputed. Do not claim the application proves discrimination or is ready for lending decisions.")
    st.markdown("**SHAP is explanatory, not causal:** it describes this fitted model relative to its tree-based reference. Correlated features and alternative reference choices can affect attribution.")
    st.dataframe(read("training_weights.csv"), hide_index=True, width="stretch")
    with st.expander("Reproducibility and limitations"):
        st.json(manifest)
    st.markdown("[UCI German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) · [Coding correction](https://archive.ics.uci.edu/dataset/573/south+german+credit+update) · [Fairlearn](https://fairlearn.org/) · [SHAP](https://shap.readthedocs.io/)")
    st.download_button("Download evaluation metadata", json.dumps(manifest, indent=2), "manifest.json", "application/json")

st.divider()
st.caption("LOAN APPROVAL BIAS AUDITOR · Educational credit-risk proxy study · Local, reproducible, and transparent")
