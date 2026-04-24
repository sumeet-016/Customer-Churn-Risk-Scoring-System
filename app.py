import sys
import os

# ✅ Fix import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Risk System",
    page_icon="📊",
    layout="wide"
)

# ── Cache Model Loading — only loads pkl files ONCE ───────────────────────────
@st.cache_resource
def load_pipeline():
    return PredictPipeline()

pipeline = load_pipeline()  # ✅ loaded once, reused on every prediction

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Risk Scoring System")
st.markdown("""
This system identifies high-risk customers using a tuned Ensemble Model.
Optimized for **Recall** to ensure potential churners are proactively flagged.
""")

# ── Sidebar ROI Calculator ─────────────────────────────────────────────────────
st.sidebar.header("💰 Retention ROI Calculator")
incentive_cost = st.sidebar.number_input("Incentive Cost ($)",      value=60)
avg_revenue    = st.sidebar.number_input("Avg Monthly Revenue ($)", value=70)
net_gain       = (avg_revenue * 12) - incentive_cost
st.sidebar.metric("Projected 12-Month Value Saved", f"${net_gain:,}")

# ── Input Form ─────────────────────────────────────────────────────────────────
with st.form("churn_input_form"):
    st.subheader("Customer Profile")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender           = st.selectbox("Gender",            ["Female", "Male"])
        senior_citizen   = st.selectbox("Senior Citizen",    [0, 1])
        partner          = st.selectbox("Partner",           ["Yes", "No"])
        dependents       = st.selectbox("Dependents",        ["Yes", "No"])
        tenure           = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=12)
        phone_service    = st.selectbox("Phone Service",     ["Yes", "No"])
        multiple_lines   = st.selectbox("Multiple Lines",    ["No phone service", "No", "Yes"])

    with col2:
        internet_service  = st.selectbox("Internet Service",  ["Fiber optic", "DSL", "No"])
        online_security   = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
        online_backup     = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support      = st.selectbox("Tech Support",      ["No", "Yes", "No internet service"])
        streaming_tv      = st.selectbox("Streaming TV",      ["No", "Yes", "No internet service"])
        streaming_movies  = st.selectbox("Streaming Movies",  ["No", "Yes", "No internet service"])

    with col3:
        contract          = st.selectbox("Contract",          ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method    = st.selectbox("Payment Method",    [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        m_charges = st.number_input("Monthly Charges ($)", min_value=0.0,  max_value=200.0, value=50.0)
        t_charges = st.number_input("Total Charges ($)",   min_value=0.0, max_value=10000.0, value=500.0)

    submitted = st.form_submit_button("🔍 Predict Churn Risk")

# ── Prediction ─────────────────────────────────────────────────────────────────
if submitted:
    try:
        # Step 1 — Build input DataFrame
        custom_data = CustomData(
            gender=gender, SeniorCitizen=senior_citizen,
            Partner=partner, Dependents=dependents,
            tenure=tenure, PhoneService=phone_service,
            MultipleLines=multiple_lines, InternetService=internet_service,
            OnlineSecurity=online_security, OnlineBackup=online_backup,
            DeviceProtection=device_protection, TechSupport=tech_support,
            StreamingTV=streaming_tv, StreamingMovies=streaming_movies,
            Contract=contract, PaperlessBilling=paperless_billing,
            PaymentMethod=payment_method,
            MonthlyCharges=m_charges, TotalCharges=t_charges
        )

        input_df = custom_data.get_data_as_data_frame()

        # Step 2 — Predict
        # ✅ Unpack both prediction and probability
        prediction, probability = pipeline.predict(input_df)

        # Step 3 — Display Results
        st.divider()
        st.subheader("Prediction Result")

        col_result, col_prob = st.columns(2)

        with col_result:
            if prediction == 1:
                st.error("🚨 HIGH CHURN RISK")
                st.write("This customer matches the high-risk profile.")
                st.info(f"💡 **Recommended Action** — Offer retention incentive.\n\n"
                        f"Projected 12-month value saved: **${net_gain:,}**")
            else:
                st.success("✅ LOW CHURN RISK")
                st.write("Customer behavior suggests high loyalty.")

        with col_prob:
            # ✅ Show churn probability as a metric and progress bar
            st.metric(
                label="Churn Probability",
                value=f"{probability}%",
                delta="High Risk" if prediction == 1 else "Low Risk"
            )
            st.progress(int(probability))  # visual risk bar

            # Risk tier
            if probability >= 70:
                st.error("Risk Tier: 🔴 Critical")
            elif probability >= 50:
                st.warning("Risk Tier: 🟠 High")
            elif probability >= 35:
                st.warning("Risk Tier: 🟡 Medium")
            else:
                st.success("Risk Tier: 🟢 Low")

    except Exception as e:
        st.error(f"Prediction failed: {e}")