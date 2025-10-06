import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import xgboost as xgb

# --------------------------------------------
# 💾 Load trained model
# --------------------------------------------
model = joblib.load("loan_default_model.pkl")

# --------------------------------------------
# 🧭 Page setup
# --------------------------------------------
st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="💰", layout="centered")
st.title("💰 Loan Default Risk Predictor")
st.markdown(
    """
    Predict whether a borrower is likely to default using a trained **XGBoost model**.
    Explore how each factor contributes using **SHAP Explainability**.
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
# 🧮 Feature Engineering (same as training)
# --------------------------------------------
debt_to_income = loan_amount / (income + 1e-6)
credit_utilization = loan_amount / (credit_score + 1e-6)
monthly_installment = loan_amount / (loan_term + 1e-6)

# Combine features
features = np.array([[
    age, income, loan_amount, credit_score, months_employed,
    num_credit_lines, interest_rate, loan_term, dti_ratio,
    debt_to_income, credit_utilization, monthly_installment
]])

columns = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed",
    "NumCreditLines", "InterestRate", "LoanTerm", "DTIRatio",
    "DebtToIncomeRatio", "CreditUtilization", "MonthlyInstallment"
]

input_df = pd.DataFrame(features, columns=columns)

# --------------------------------------------
# 🤖 Model Prediction
# --------------------------------------------
pred_prob = model.predict_proba(input_df)[:, 1][0]
pred_class = int(model.predict(input_df)[0])

st.subheader("🔍 Prediction Result")
st.write(f"**Predicted Default Probability:** {pred_prob:.2%}")

if pred_prob < 0.3:
    st.success("🟢 Low Risk — Borrower is unlikely to default.")
elif pred_prob < 0.7:
    st.warning("🟡 Medium Risk — Borrower has moderate default probability.")
else:
    st.error("🔴 High Risk — Borrower is likely to default!")

# --------------------------------------------
# 📊 Explainability with SHAP
# --------------------------------------------
with st.expander("📊 Explain Model Prediction (SHAP)", expanded=False):
    explainer = shap.Explainer(model)
    shap_values = explainer(input_df)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.markdown("### Feature Impact on Prediction")
    shap.plots.bar(shap_values)
    st.pyplot()
