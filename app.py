import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved model
with open('best_rf_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# Title and intro
st.title("💓 Heart Disease Risk Prediction App")
st.markdown("""
Welcome to the **Heart Disease Prediction App**.  
This tool uses a machine learning model (Random Forest Classifier) trained on medical data to predict the likelihood of heart disease.
""")

# Sidebar for inputs
st.sidebar.header("🧾 Patient Input Features")
st.sidebar.markdown("Please provide the following details:")

# Sidebar inputs
age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
cp = st.sidebar.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [1, 0])
restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 210, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
slope = st.sidebar.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels Colored", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Input array
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])

columns = ["Age", "Sex", "Chest Pain", "RestBP", "Chol", "FBS", "RestECG", 
           "MaxHR", "ExAng", "Oldpeak", "Slope", "CA", "Thal"]

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader("🔎 Prediction Result")
    if prediction == 1:
        st.error("⚠️ **High Risk of Heart Disease!**")
    else:
        st.success("✅ **Low Risk of Heart Disease.**")

    st.markdown(f"**Prediction Confidence:**")
    st.progress(round(probabilities[prediction]*100))

    # Show input summary
    st.subheader("📋 Patient Summary")
    st.table(pd.DataFrame(input_data, columns=columns))

    # Feature contribution approximation (using feature importances)
    st.subheader("📊 Feature Influence (Relative Importance)")
    importance = model.feature_importances_
    sorted_indices = np.argsort(importance)[::-1]
    
    fig, ax = plt.subplots()
    ax.barh(np.array(columns)[sorted_indices], importance[sorted_indices], color='skyblue')
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Influences (Random Forest)")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("🔬 Model trained using Random Forest | Made by Fatima Azfar | Data Source: UCI Heart Disease Dataset")
