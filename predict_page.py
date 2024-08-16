import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

logistic = data["model"]
le_sex = data["le_sex"]

def show_predict_page():
    st.title("Diabetes Prediction")

    st.write("""We need some information to predict your diabetes""")
    
    sex = (
        "Female",
        "Male"
    )
    heart_disease = (0,1)
    hypertension = (0,1)
    sex = st.selectbox("Sex",sex)
    age = st.slider("Age",0,100)
    hypertension =st.selectbox("hypertension",hypertension)
    heart_disease =st.selectbox("heart_disease",heart_disease)
    bmi = st.number_input("BMI",0.0)
    blood_glucose_level = st.number_input("Blood Glucose Level",0.0)
    HbA1c_level = st.number_input("HbA1c",0.0)
    ok = st.button("Calculate Diabetes")
    if ok:
        X = np.array([[sex,age,hypertension,heart_disease,bmi,HbA1c_level,blood_glucose_level]])
        X[:, 0] = le_sex.transform(X[:,0])
        X = X.astype(float)

        diabetes = logistic.predict(X)
        if diabetes == 1:
            st.subheader("You have diabetes!")
        else:
            st.subheader("You don't have diabetes")

show_predict_page()