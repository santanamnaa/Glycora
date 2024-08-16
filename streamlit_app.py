import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

# Load model function
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
logistic = data["model"]
le_sex = data["le_sex"]

# Load and clean the data for the explore page
@st.cache_data
def load_data():
    diabetes = pd.read_csv("diabetes.csv")
    diabetes.drop('smoking_history', axis=1, inplace=True)
    diabetes['gender'] = le_sex.fit_transform(diabetes['gender'])
    
    Q1 = np.percentile(diabetes['bmi'], 25)
    Q3 = np.percentile(diabetes['bmi'], 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    diabetes_cleaned = diabetes[(diabetes['bmi'] >= lower_bound) & (diabetes['bmi'] <= upper_bound)]
    
    return diabetes_cleaned

# Explore page content
def show_explore_page():
    st.title("Explore Diabetes Data")
    df = load_data()

    X = df.drop('diabetes', axis=1)
    y = df['diabetes']

    y_pred = logistic.predict(X)
    cm = confusion_matrix(y, y_pred)
    cm_labels = ['No Diabetes', 'Diabetes']

    fig_cm = ff.create_annotated_heatmap(z=cm, x=cm_labels, y=cm_labels, colorscale='Blues', showscale=True, annotation_text=cm.astype(str))
    fig_cm.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')
    
    st.write("Confusion Matrix")
    st.plotly_chart(fig_cm)

# Predict page content
def show_predict_page():
    st.title("Glycora (Diabetes prediction)")

    st.write("We need some information to predict your diabetes.")
    
    sex_options = ("Female", "Male")
    sex = st.selectbox("Sex", sex_options, key="predict_sex")
    
    age = st.slider("Age", 0, 100, key="predict_age")
    hypertension = st.selectbox("Hypertension (0: No, 1: Yes)", (0, 1), key="predict_hypertension")
    heart_disease = st.selectbox("Heart Disease (0: No, 1: Yes)", (0, 1), key="predict_heart_disease")
    bmi = st.number_input("BMI", 0.0, key="predict_bmi")
    blood_glucose_level = st.number_input("Blood Glucose Level", 0.0, key="predict_glucose")
    HbA1c_level = st.number_input("HbA1c", 0.0, key="predict_hba1c")

    ok = st.button("Calculate Diabetes", key="predict_button")
    
    if ok:
        X = np.array([[sex, age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level]])
        X[:, 0] = le_sex.transform(X[:, 0])
        X = X.astype(float)

        diabetes = logistic.predict(X)
        if diabetes == 1:
            st.subheader("You have diabetes!")
        else:
            st.subheader("You don't have diabetes.")

# Main app to switch between Explore and Predict pages
page = st.sidebar.selectbox("Choose a page", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
else:
    show_explore_page()
