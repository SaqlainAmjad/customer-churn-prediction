import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load models safely
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_churn_model.pkl")
    preprocessor = joblib.load("models/preprocessor.pkl")
    feature_names = joblib.load("models/feature_names.pkl")
    return model, preprocessor, feature_names


model, preprocessor, feature_names = load_artifacts()

# -----------------------------
# App UI
# -----------------------------
st.title("📊 Telco Customer Churn Prediction")
st.write("Enter customer details to predict churn probability.")

st.sidebar.header("Customer Information")

# -----------------------------
# Input Function
# -----------------------------
def user_input_features():
    data = {
        "gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
        "SeniorCitizen": st.sidebar.selectbox("Senior Citizen", [0, 1]),
        "Partner": st.sidebar.selectbox("Partner", ["Yes", "No"]),
        "Dependents": st.sidebar.selectbox("Dependents", ["Yes", "No"]),
        "tenure": st.sidebar.slider("Tenure (months)", 1, 72, 12),
        "PhoneService": st.sidebar.selectbox("Phone Service", ["Yes", "No"]),
        "MultipleLines": st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"]),
        "InternetService": st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"]),
        "OnlineBackup": st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"]),
        "DeviceProtection": st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"]),
        "TechSupport": st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"]),
        "StreamingTV": st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"]),
        "StreamingMovies": st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"]),
        "Contract": st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": st.sidebar.selectbox("Paperless Billing", ["Yes", "No"]),
        "PaymentMethod": st.sidebar.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        ),
        "MonthlyCharges": st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0),
        "TotalCharges": st.sidebar.number_input("Total Charges", 0.0, 10000.0, 500.0),
    }

    return pd.DataFrame([data])


input_df = user_input_features()

# -----------------------------
# Fix inconsistent values
# -----------------------------
for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"]:
    input_df[col] = input_df[col].replace({
        "No internet service": "No",
        "No phone service": "No"
    })

# Ensure numeric type
input_df["TotalCharges"] = pd.to_numeric(input_df["TotalCharges"], errors="coerce").fillna(0)

# -----------------------------
# Align columns (IMPORTANT FIX)
# -----------------------------
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# -----------------------------
# Preprocess + Predict
# -----------------------------
processed_data = preprocessor.transform(input_df)

st.subheader("User Input")
st.write(input_df)

if st.button("Predict Churn"):
    proba = model.predict_proba(processed_data)[0][1]
    prediction = 1 if proba > 0.5 else 0

    st.subheader("Prediction Result")

    st.write(f"📊 Churn Probability: {proba:.2f}")

    if prediction == 1:
        st.error("⚠️ This customer is likely to CHURN")
    else:
        st.success("✅ This customer is NOT likely to churn")
