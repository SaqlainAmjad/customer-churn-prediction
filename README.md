# 🚀 Customer Churn Prediction

A machine learning project that predicts whether a customer is likely to churn based on historical data. This project covers the full pipeline from data analysis to model deployment with an interactive Streamlit web app.

---

## 📌 Problem Statement

Customer churn is a critical challenge for subscription-based businesses. Identifying customers who are likely to leave helps companies take proactive actions to improve retention and increase revenue.

---

## 🧠 Solution

This project builds a predictive model using machine learning techniques to classify customers as **churn** or **non-churn** based on their behavior and attributes.

---

## ⚙️ Tech Stack

- Python  
- Pandas & NumPy  
- Scikit-learn  
- Matplotlib & Seaborn  
- Streamlit  

---

## 📊 Project Workflow

1. **Data Preprocessing**
   - Handling missing values  
   - Encoding categorical variables  
   - Feature scaling  

2. **Exploratory Data Analysis (EDA)**
   - Understanding customer behavior  
   - Identifying key churn indicators  

3. **Feature Engineering**
   - Selecting important features  
   - Creating meaningful transformations  

4. **Model Building**
   - Training multiple ML models  
   - Selecting the best-performing model  

5. **Model Evaluation**
   - Accuracy, Precision, Recall  
   - ROC-AUC Score  

6. **Deployment**
   - Streamlit app for real-time predictions  

---

## 📂 Project Structure
customer-churn-prediction/
│
├── notebooks/ # Jupyter notebook for analysis & modeling
├── app/ # Streamlit application
├── models/ # Saved model & preprocessing files
├── data/ # Sample dataset (optional)
├── requirements.txt # Dependencies
├── README.md # Project documentation

## 🖥️ Run Locally

Clone the repository:

```bash
git clone https://github.com/SaqlainAmjad/customer-churn-prediction.git
cd customer-churn-prediction
Install dependencies:
pip install -r requirements.txt
Run the Streamlit app:
streamlit run app/streamlit_app.py

## Key Insights
Customers with higher monthly charges are more likely to churn
Long-term customers are less likely to leave
Certain service features significantly impact churn behavior

## Acknowledgements
This project was built as part of a hands-on learning approach to machine learning and real-world problem solving.
