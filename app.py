import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ===== PATHS (local folder) =====
MODEL_DIR = "."
DATA_DIR = "."

# ===== LOAD MODELS =====
model_treated = joblib.load(f"{MODEL_DIR}/model_treated.pkl")
model_control = joblib.load(f"{MODEL_DIR}/model_control.pkl")

st.set_page_config(page_title="Customer Uplift Dashboard", layout="wide")
st.title("📊 Customer Uplift Decision Dashboard")

uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file, sep=";")
    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    # ===== PREPARE DATA =====
    data = data.copy()

    if "contact" in data.columns:
        data["treatment"] = data["contact"].apply(lambda x: 1 if x != "unknown" else 0)
    else:
        st.error("Missing required column: contact")
        st.stop()

    if "y" in data.columns:
        data["outcome"] = data["y"].apply(lambda x: 1 if x == "yes" else 0)
    else:
        data["outcome"] = 0

    if "duration" in data.columns:
        data = data.drop(columns=["duration"])

    # ===== FEATURE MATRIX =====
    X = data.drop(columns=[c for c in ["y", "outcome", "treatment"] if c in data.columns])
    X = pd.get_dummies(X, drop_first=True)

    # Align columns with training
    X = X.reindex(columns=model_treated.feature_names_in_, fill_value=0)

    # ===== PREDICTION =====
    p_treated = model_treated.predict_proba(X)[:, 1]
    p_control = model_control.predict_proba(X)[:, 1]
    uplift = p_treated - p_control

    results = data.copy()
    results["uplift"] = uplift

    # ===== BUSINESS LOGIC =====
    if "balance" not in results.columns:
        st.error("Missing required column: balance")
        st.stop()

    results["deposit_amount"] = np.maximum(1000, results["balance"] * np.random.uniform(0.1, 0.5))
    results["value"] = results["deposit_amount"] * 0.03 * 1.5

    cost_map = {"cellular": 30, "telephone": 50, "unknown": 5}
    results["contact_cost"] = results["contact"].map(cost_map).fillna(10)

    results["expected_profit"] = results["uplift"] * results["value"] - results["contact_cost"]

    threshold = results["uplift"].quantile(0.8)
    results["recommend_contact"] = results["uplift"] > threshold

    # ===== DISPLAY =====
    st.subheader("🎯 Top Customers to Focus On")
    st.dataframe(results.sort_values("uplift", ascending=False).head(20))

    st.subheader("📈 Summary Metrics")
    col1, col2, col3 = st.columns(3)

    col1.metric("Mean Uplift", round(results["uplift"].mean(), 4))
    col2.metric("Recommend Rate", round(results["recommend_contact"].mean(), 4))
    col3.metric("Mean Profit (Targeted)", round(results.loc[results["recommend_contact"], "expected_profit"].mean(), 2))

    st.subheader("⬇ Download Recommendations")
    csv = results.sort_values("uplift", ascending=False).to_csv(index=False).encode("utf-8")
    st.download_button("Download Results CSV", csv, "uplift_results.csv", "text/csv")
