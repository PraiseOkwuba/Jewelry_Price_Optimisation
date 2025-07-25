import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import lightgbm as lgb
import matplotlib.pyplot as plt

# --- Load model and SHAP explainer ---
@st.cache_resource
def load_model():
    model = joblib.load("lgbm_jewelry_model.pkl")
    return model

@st.cache_resource
def get_explainer(_model):
    return shap.Explainer(_model)

model = load_model()
explainer = get_explainer(model)

# --- Define feature schema ---
features = [
    "carat", "cut_quality", "clarity", "color_grade",
    "depth_pct", "table_pct", "cert_lab", "shape",
    "polish", "symmetry", "fluorescence"
]
cat_cols = ["cut_quality", "clarity", "color_grade", "cert_lab", "shape", "polish", "symmetry", "fluorescence"]

# --- UI: Title ---
st.title("💎 Jewelry Price Predictor (Optimized LightGBM)")
st.write("Enter gemstone characteristics below to estimate its price per unit (USD).")

# --- UI: Input form ---
with st.form("input_form"):
    input_data = {}
    input_data["carat"] = st.slider("Carat", 0.1, 5.0, 1.0, 0.01)
    input_data["depth_pct"] = st.slider("Depth (%)", 50.0, 70.0, 62.0)
    input_data["table_pct"] = st.slider("Table (%)", 50.0, 70.0, 57.0)
    input_data["cut_quality"] = st.selectbox("Cut Quality", ["Ideal", "Very Good", "Good", "Fair"])
    input_data["clarity"] = st.selectbox("Clarity", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2"])
    input_data["color_grade"] = st.selectbox("Color Grade", ["D", "E", "F", "G", "H", "I", "J"])
    input_data["cert_lab"] = st.selectbox("Certification Lab", ["GIA", "IGI", "HRD", "None"])
    input_data["shape"] = st.selectbox("Shape", ["Round", "Princess", "Emerald", "Oval", "Cushion"])
    input_data["polish"] = st.selectbox("Polish", ["Excellent", "Very Good", "Good"])
    input_data["symmetry"] = st.selectbox("Symmetry", ["Excellent", "Very Good", "Good"])
    input_data["fluorescence"] = st.selectbox("Fluorescence", ["None", "Faint", "Medium", "Strong"])

    submitted = st.form_submit_button("💰 Predict Price")

# --- Prediction logic ---
if submitted:
    df_input = pd.DataFrame([input_data])

    # Ensure correct dtype for categorical columns
    for col in cat_cols:
        df_input[col] = df_input[col].astype("category")

    log_pred = model.predict(df_input)
    price = np.exp(log_pred[0])

    st.subheader(f"💸 Predicted Price per Unit: **${price:,.2f}**")

    # SHAP Explanation
    st.subheader("🔍 Model Explanation (SHAP)")
    shap_values = explainer(df_input)
    st.set_option("deprecation.showPyplotGlobalUse", False)
    shap.plots.waterfall(shap_values[0])
    st.pyplot(bbox_inches='tight')

    st.caption("Model: LightGBM (Optuna Tuned) | Output: Price per unit in USD")
