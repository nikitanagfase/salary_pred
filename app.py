import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Salary Predictor", layout="centered")

st.title("💼 Salary Prediction App")

# ---------------- Load model ----------------
MODEL_PATH = "best_salary_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

if model is None:
    st.error("❌ Model file not found in repository")
    st.stop()

# ---------------- Inputs ----------------
age = st.slider("Age", 18, 65, 30)

gender = st.radio("Gender", ["Female", "Male"])
gender = 1 if gender == "Male" else 0

education = st.selectbox("Education", ["Bachelor's", "Master's", "PhD"])
education = {"Bachelor's": 0, "Master's": 1, "PhD": 2}[education]

job_titles = {
    "Software Engineer": 10,
    "Data Scientist": 20,
    "Manager": 30,
    "Business Analyst": 40,
}

job = st.selectbox("Job Title", list(job_titles.keys()))
job = job_titles[job]

experience = st.slider("Experience (years)", 0.0, 40.0, 5.0)

# ---------------- Predict ----------------
if st.button("Predict Salary"):
    try:
        X = np.array([[age, gender, education, job, experience]])
        pred = model.predict(X)

        st.success(f"💰 Estimated Salary: ${pred[0]:,.2f}")

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))
