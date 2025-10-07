# ----------------------------------------------------------
# 💰 Loan Default Risk Predictor (Fixed SHAP Warning Version)
# ----------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings

# --------------------------------------------
# ⚙️ Suppress Warnings (Optional)
# --------------------------------------------
warnings.filterwarnings("ignore")

# --------------------------------------------
# 💾 Load Model and Scaler
# --------------------------------------------
model = joblib.load("loan_default_model.pkl")
scaler = joblib.load("scaler.pkl")

# --------------------------------------------
# 🧭 Page Setup
# --------------------------------------------
st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="💰", layout="centered")
st.title("💰 Loan Default Risk Predictor")
st.markdown(
    """
    Predict whether a borrower is likely to **default** using a trained **XGBoost model**.  
    Understand feature impact with **SHAP Explainability**.
    """
)

# --------------------------------------------
# 📋 Sidebar: Borrower Inputs
# --------------------------------------------
st.sidebar.header("📋 Borrower Details")

age = st.sidebar.slider("Age", 18, 80, 35)
income = st.sidebar.number_input("Annual Income ($)", 1000, 200000, 50000)
loan_amount = st.sidebar.number_input("Loan Amount ($)", 1000, 200000, 20000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
months_employed = st.sidebar.slider("Months Employed", 0, 360, 60)
num_credit_lines = st.sidebar.slider("Number of Credit Lines", 0, 20, 3)
interest_rate = st.sidebar.slider("Interest Rate (%)", 0.0, 30.0, 10.0)
loan_term = st.sidebar.slider("Loan Term (months)", 6, 84, 36)
dti_ratio = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 2.0, 0.4)

# --------------------------------------------
# 🧮 Feature Engineering (Same as Training)
# --------------------------------------------
debt_to_income = loan_amount / (income + 1e-6)
credit_utilization = loan_amount / (credit_score + 1e-6)
monthly_installment = loan_amount / (loan_term + 1e-6)

# Combine features
features = np.array([[age, income, loan_amount, credit_score, months_employed,
                      num_credit_lines, interest_rate, loan_term, dti_ratio,
                      debt_to_income, credit_utilization, monthly_installment]])

columns = ["Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
           "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
           "DebtToIncomeRatio", "CreditUtilization", "MonthlyInstallment"]

input_df = pd.DataFrame(features, columns=columns)

# Scale features
scaled_input = scaler.transform(input_df)

# --------------------------------------------
# 🤖 Model Prediction
# --------------------------------------------
pred_prob = model.predict_proba(scaled_input)[:, 1][0]
pred_class = int(model.predict(scaled_input)[0])

st.subheader("🔍 Prediction Result")
st.write(f"**Predicted Default Probability:** {pred_prob:.2%}")

if pred_prob < 0.3:
    st.success("🟢 Low Risk — Borrower is unlikely to default.")
elif pred_prob < 0.7:
    st.warning("🟡 Medium Risk — Borrower has moderate default probability.")
else:
    st.error("🔴 High Risk — Borrower is likely to default!")

# --------------------------------------------
# 📊 Explainability with SHAP (Fixed)
# --------------------------------------------
with st.expander("📊 Explain Model Prediction (SHAP)", expanded=False):
    st.markdown("### Feature Impact on Prediction")

    # Initialize SHAP Explainer and compute values
    explainer = shap.Explainer(model)
    shap_values = explainer(scaled_input)

    # ✅ Fix: Explicit figure to avoid Streamlit warning
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values, show=False)
    st.pyplot(fig, bbox_inches='tight')
    plt.close(fig)

    # Optional: Show detailed summary plot (dot plot)
    st.markdown("### Detailed SHAP Summary")
    fig2, ax2 = plt.subplots()
    shap.summary_plot(shap_values, input_df, show=False)
    st.pyplot(fig2, bbox_inches='tight')
    plt.close(fig2)
