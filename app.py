import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title("💼 Salary Prediction App")
st.write("Enter employee details to predict estimated salary.")

# ----------------------------
# Load model safely
# ----------------------------
MODEL_PATH = "best_salary_model.pkl"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

if model is None:
    st.error("❌ Model file not found. Please check repository upload.")
    st.stop()

# ----------------------------
# Input fields
# ----------------------------
age = st.slider("Age", 18, 65, 30)

gender = st.radio("Gender", ["Female", "Male"])
gender = 1 if gender == "Male" else 0

education = st.selectbox("Education Level", ["Bachelor's", "Master's", "PhD"])
education_map = {"Bachelor's": 0, "Master's": 1, "PhD": 2}
education = education_map[education]

job_titles = {
    "Software Engineer": 10,
    "Data Scientist": 20,
    "Manager": 30,
    "Business Analyst": 40,
    "HR Specialist": 50
}

job_title = st.selectbox("Job Title", list(job_titles.keys()))
job_title = job_titles[job_title]

experience = st.slider("Years of Experience", 0.0, 40.0, 5.0)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Salary"):
    try:
        features = np.array([[age, gender, education, job_title, experience]])
        prediction = model.predict(features)

        st.success(f"💰 Estimated Salary: ${prediction[0]:,.2f}")

    except Exception as e:
        st.error("❌ Prediction failed. Check model compatibility.")
        st.code(str(e))
