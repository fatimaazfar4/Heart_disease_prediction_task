# 💓 Heart Disease Prediction Using Machine Learning

A machine learning project that predicts the likelihood of heart disease based on patient data. This project compares the performance of **Logistic Regression** and **Random Forest**, with the best model deployed via a **Streamlit** web app.

---

## 📁 Project Files

- `heart_disease.csv` – Raw dataset with 1025 rows and 14 columns  
- `heart_disease.ipynb` – Notebook for data preprocessing, model training, and evaluation  
- `best_rf_model.pkl` – Trained Random Forest model (best performer)  
- `app.py` – Streamlit web app for interactive predictions  
- `requirements.txt` – List of required Python libraries  
- `README.md` – Project documentation

---

## 📊 Dataset Overview

- **Source:** UCI Heart Disease Dataset  
- **Total entries:** 1025  
- **After removing duplicates:** 302  
- **Target Column:** `target` (1 = has disease, 0 = no disease)  
- **Key Features:**  
  - Demographic: `age`, `sex`  
  - Medical: `cp`, `chol`, `thalach`, `oldpeak`, etc.

---

## 🧠 Machine Learning Models

### 1. Logistic Regression

- **Purpose:** Used as a baseline classification model.
- **Steps:**
  - Trained on the cleaned dataset.
  - Evaluated using accuracy, confusion matrix, and classification report.
- **Outcome:** Reasonable performance, but not the best.

### 2. Random Forest Classifier (Chosen Model ✅)

- **Why Random Forest?** It gave higher accuracy and handled feature interactions better.
- **Steps:**
  - Performed **hyperparameter tuning** using `GridSearchCV`.
  - Best parameters (e.g. `n_estimators`, `max_depth`, etc.) were selected.
  - Trained the final model on the full cleaned dataset.
  - Saved using `pickle` as `best_rf_model.pkl` for later use in deployment.

---

## 🧪 Model Evaluation

Both models were evaluated using:
- **Accuracy Score**
- **Confusion Matrix**
- **Precision, Recall, F1-Score**

🔍 Final comparison showed **Random Forest outperformed Logistic Regression**, especially in correctly identifying true positives (patients with disease).

---

## 🚀 Web App (Streamlit)

A lightweight Streamlit app (`app.py`) was built to:
- Take user inputs for all required medical features
- Load the saved Random Forest model (`.pkl`)
- Display prediction result instantly (At Risk / Not at Risk)

To run the app:
```bash
streamlit run app.py
