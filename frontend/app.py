import streamlit as st
import requests
import pandas as pd

feature_importance = pd.read_csv("models/feature_importance.csv")

st.title("Loan Approval Prediction System")

st.write("Enter applicant details to check loan approval.")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input("Applicant Income")
coapplicant_income = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Amount Term")
credit_history = st.selectbox("Credit History", [1.0, 0.0])

property_area = st.selectbox(
    "Property Area",
    ["Urban", "Semiurban", "Rural"]
)
st.subheader("Feature Importance")

st.bar_chart(
    feature_importance.set_index("Feature")
)

if st.button("Predict Loan Status"):

    input_data = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=input_data
    )

    result = response.json()

    st.subheader("Prediction Result")
    st.success(result["prediction"])

    st.subheader("Approval Probability")

    prob = result["approval_probability"]

    st.progress(prob)
    st.write(f"Approval Probability: {prob*100:.2f}%")