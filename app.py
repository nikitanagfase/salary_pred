import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Load the saved model
with open('best_salary_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Salary Prediction App")
st.write("Enter details to predict the estimated salary.")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: 'Male' if x==1 else 'Female')
education = st.selectbox("Education Level", options=[0, 1, 2], format_func=lambda x: ["Bachelor's", "Master's", "PhD"][x])
job_title = st.number_input("Job Title (Encoded ID)", min_value=0, max_value=200, value=100)
exp = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=5.0)

if st.button("Predict Salary"):
    features = np.array([[age, gender, education, job_title, exp]])
    prediction = model.predict(features)
    st.success(f"The estimated salary is: ${prediction[0]:,.2f}")
