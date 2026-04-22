


import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_artifacts():
    best_model = joblib.load("best_churn_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return best_model, preprocessor, feature_names

best_model, preprocessor, feature_names = load_artifacts()

# --- Streamlit UI ---
st.title("Telco Customer Churn Prediction")
st.write("Predict if a customer will churn based on their service details.")

# --- Input Features ---
st.sidebar.header("Customer Information")

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    SeniorCitizen = st.sidebar.selectbox('Senior Citizen', (0, 1), format_func=lambda x: 'Yes' if x == 1 else 'No')
    Partner = st.sidebar.selectbox('Partner', ('Yes', 'No'))
    Dependents = st.sidebar.selectbox('Dependents', ('Yes', 'No'))
    tenure = st.sidebar.slider('Tenure (months)', 1, 72, 12)
    PhoneService = st.sidebar.selectbox('Phone Service', ('Yes', 'No'))
    MultipleLines = st.sidebar.selectbox('Multiple Lines', ('No phone service', 'No', 'Yes'))
    InternetService = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    OnlineSecurity = st.sidebar.selectbox('Online Security', ('No', 'Yes', 'No internet service'))
    OnlineBackup = st.sidebar.selectbox('Online Backup', ('No', 'Yes', 'No internet service'))
    DeviceProtection = st.sidebar.selectbox('Device Protection', ('No', 'Yes', 'No internet service'))
    TechSupport = st.sidebar.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
    StreamingTV = st.sidebar.selectbox('Streaming TV', ('No', 'Yes', 'No internet service'))
    StreamingMovies = st.sidebar.selectbox('Streaming Movies', ('No', 'Yes', 'No internet service'))
    Contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    PaperlessBilling = st.sidebar.selectbox('Paperless Billing', ('Yes', 'No'))
    PaymentMethod = st.sidebar.selectbox('Payment Method',
                                        ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    MonthlyCharges = st.sidebar.slider('Monthly Charges', 18.0, 118.0, 70.0)
    TotalCharges = st.sidebar.slider('Total Charges', 0.0, 8684.0, 500.0)

    data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# --- Preprocessing Input ---
# Ensure columns are in the correct order for the preprocessor and handle formatting
# The preprocessor expects columns as they were in X_train originally
original_X_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                   'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                   'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                   'MonthlyCharges', 'TotalCharges']

# Create a DataFrame with all original columns, filled with defaults, then update with user input
processed_input_df = pd.DataFrame(columns=original_X_cols)
for col in original_X_cols:
    if col in input_df.columns:
        processed_input_df[col] = input_df[col]
    else:
        # Default values for columns not directly in input_df (shouldn't happen with full UI)
        if col in ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']:
            processed_input_df[col] = 0.0
        else:
            processed_input_df[col] = 'No' # Adjust as per your data's typical 'No' equivalent

# Apply standardization to categorical columns that were adjusted in data cleaning
for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
            'StreamingTV', 'StreamingMovies', 'MultipleLines']:
    processed_input_df[col] = processed_input_df[col].replace({'No internet service': 'No', 'No phone service': 'No'})

# Convert TotalCharges to numeric, handling potential errors (though slider should prevent)
processed_input_df['TotalCharges'] = pd.to_numeric(processed_input_df['TotalCharges'], errors='coerce').fillna(0)

# Preprocess the data using the loaded preprocessor
processed_data = preprocessor.transform(processed_input_df)
processed_data_df = pd.DataFrame(processed_data, columns=feature_names)


st.subheader('User Input Features')
st.write(input_df)

st.subheader('Prediction')
if st.button('Predict Churn'):
    prediction_proba = best_model.predict_proba(processed_data_df)[:, 1]
    prediction_label = (prediction_proba > 0.5).astype(int)

    st.write(f"Churn Probability: {prediction_proba[0]:.2f}")
    st.write(f"Predicted Churn: {'Yes' if prediction_label[0] == 1 else 'No'}")

    if prediction_label[0] == 1:
        st.error("This customer is predicted to CHURN!")
    else:
        st.success("This customer is predicted NOT to churn.")

st.markdown("""
**How to run this Streamlit app:**
1. Save the code above into a Python file, e.g., `streamlit_app.py`.
2. Ensure you have Streamlit installed: `pip install streamlit`.
3. Make sure `best_churn_model.pkl`, `preprocessor.pkl`, and `feature_names.pkl` are in the same directory.
4. Run from your terminal: `streamlit run streamlit_app.py`
""")
