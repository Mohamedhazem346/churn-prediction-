import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Title
# -----------------------------

st.markdown(
    "<h1 style='text-align: center;'>📊 Customer Churn Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown("Machine Learning Model using **Naive Bayes** 🤖")

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_excel("churn_dataset.xlsx")

st.subheader("Dataset Preview")
st.dataframe(df.head(7))

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("model_chrun.pkl") 

# -----------------------------
# Sidebar Inputs
# -----------------------------

st.sidebar.title("Customer Information")

age = st.sidebar.slider(
    "Age",
    int(df['Age'].min()),
    int(df['Age'].max()),
    30
)

tenure = st.sidebar.slider(
    "Tenure (Months)",
    int(df['Tenure'].min()),
    int(df['Tenure'].max()),
    12
)

gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

sex = 1 if gender == "Male" else 0

# -----------------------------
# Prediction Button
# -----------------------------

if st.sidebar.button("Predict Churn"):

    input_data = np.array([[age, tenure, sex]])

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader("Prediction Probabilities")

    labels = ["Customer Stay", "Customer Leave"]

    for label, prob in zip(labels, probabilities):
        st.write(f"{label}: {prob*100:.2f}%")
        st.progress(float(prob))

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer Will Leave")
    else:
        st.success("✅ Customer Will Stay")

# -----------------------------
# Dataset Statistics
# -----------------------------

st.subheader("Dataset Statistics")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Average Age", round(df["Age"].mean(),1))
col3.metric("Average Tenure", round(df["Tenure"].mean(),1))

# -----------------------------
# Footer
# -----------------------------

st.markdown("---")
st.caption("Developed by Mohamed Hazem | Data Mining Project") 