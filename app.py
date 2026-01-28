import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ===========================
# Load dataset & train model
# ===========================
df = pd.read_csv("C:/Users/Bhawna/Downloads/archive (4)/full_filled_stroke_data (1).csv")

# Preprocessing
df['gender'].replace({'Male': 1, 'Female': 0}, inplace=True)
df['ever_married'].replace({'Yes': 1, 'No': 0}, inplace=True)
df['Residence_type'].replace({'Urban': 1, 'Rural': 0}, inplace=True)
df = pd.get_dummies(df, columns=['work_type'], drop_first=True)
df = pd.get_dummies(df, columns=['smoking_status'], drop_first=True)

X = df.drop('stroke', axis=1)
y = df['stroke']

# Train model
model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42, class_weight='balanced')
model.fit(X, y)

feature_names = X.columns

# ===========================
# Streamlit UI
# ===========================
st.title("🧠 Stroke Risk Predictor AI")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.selectbox("Hypertension?", ["Yes", "No"])
heart_disease = st.selectbox("Heart Disease?", ["Yes", "No"])
married = st.selectbox("Married?", ["Yes", "No"])
work_type = st.selectbox("Work Type", ["Private Job", "Self Employed", "Govt Job"])
residence = st.selectbox("Residence", ["Urban", "Rural"])
avg_glucose = st.number_input("Average Glucose Level", value=100.0)
bmi = st.number_input("BMI", value=25.0)
smoking_status = st.selectbox("Smoking Status", ["Never Smoked", "Formerly Smoked", "Smokes", "Unknown"])

# ===========================
# Prediction Button
# ===========================
if st.button("Predict Risk"):
    # Create user input dataframe
    user_df = pd.DataFrame(0, index=[0], columns=feature_names)

    # Map numeric features
    user_df['gender'] = 1 if gender == 'Male' else 0
    user_df['age'] = age
    user_df['hypertension'] = 1 if hypertension == 'Yes' else 0
    user_df['heart_disease'] = 1 if heart_disease == 'Yes' else 0
    user_df['ever_married'] = 1 if married == 'Yes' else 0
    user_df['Residence_type'] = 1 if residence == 'Urban' else 0
    user_df['avg_glucose_level'] = avg_glucose
    user_df['bmi'] = bmi

    # Map work_type
    if work_type == 'Private Job' and 'work_type_Private Job' in user_df.columns:
        user_df['work_type_Private Job'] = 1
    elif work_type == 'Self Employed' and 'work_type_Self-employed' in user_df.columns:
        user_df['work_type_Self-employed'] = 1
    elif work_type == 'Govt Job' and 'work_type_Govt_Job' in user_df.columns:
        user_df['work_type_Govt_Job'] = 1

    # Map smoking_status
    if smoking_status == 'Formerly Smoked' and 'smoking_status_formerly smoked' in user_df.columns:
        user_df['smoking_status_formerly smoked'] = 1
    elif smoking_status == 'Never Smoked' and 'smoking_status_never smoked' in user_df.columns:
        user_df['smoking_status_never smoked'] = 1
    elif smoking_status == 'Smokes' and 'smoking_status_smokes' in user_df.columns:
        user_df['smoking_status_smokes'] = 1
    elif smoking_status == 'Unknown' and 'smoking_status_unknown' in user_df.columns:
        user_df['smoking_status_unknown'] = 1

    # ===========================
    # Hybrid AI Risk Logic
    # ===========================
    risk_score = 0

    # Age factor
    if age >= 60:
        risk_score += 30
    elif age >= 45:
        risk_score += 20

    # Medical conditions
    if hypertension == 'Yes':
        risk_score += 20
    if heart_disease == 'Yes':
        risk_score += 25

    # Lifestyle
    if smoking_status == 'Smokes':
        risk_score += 15
    elif smoking_status == 'Formerly Smoked':
        risk_score += 10

    # Body & glucose
    if bmi >= 30:
        risk_score += 10
    if avg_glucose >= 180:
        risk_score += 15

    # ML Prediction
    try:
        prediction = model.predict(user_df)[0]
    except:
        prediction = 0  # fallback if model fails

    # ===========================
    # Display Result
    # ===========================
    if prediction == 1 or risk_score >= 40:
        st.error("🚨 High Risk of Stroke")
    else:
        st.success("✅ Low Risk of Stroke")

    st.write(f"Risk Score: {risk_score}%")

    # Explanation
    reasons = []
    if age >= 60:
        reasons.append("Age above 60 increases risk")
    if hypertension == 'Yes':
        reasons.append("High blood pressure detected")
    if heart_disease == 'Yes':
        reasons.append("Heart disease detected")
    if smoking_status == 'Smokes':
        reasons.append("Active smoking habit")
    if avg_glucose >= 180:
        reasons.append("High glucose level")
    if bmi >= 30:
        reasons.append("Obesity detected")

    if reasons:
        st.write("⚠️ Reasons contributing to risk:")
        for r in reasons:
            st.write("- " + r)
