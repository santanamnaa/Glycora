import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import LabelEncoder
le_sex = LabelEncoder()

@st.cache_data
def load_data():

    diabetes = pd.read_csv("diabetes.csv")
    diabetes.drop('smoking_history', axis=1, inplace=True)
    diabetes['gender'] = le_sex.fit_transform(diabetes['gender'])
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = np.percentile(diabetes['bmi'], 25)
    Q3 = np.percentile(diabetes['bmi'], 75)

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the DataFrame to remove outliers
    diabetes_cleaned = diabetes[(diabetes['bmi'] >= lower_bound) & (diabetes['bmi'] <= upper_bound)]

    # Print the shape of the cleaned DataFrame to see how many outliers were removed
    return diabetes_cleaned

df = load_data()

def load_model():
    with open('saved_steps.pkl','rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

logistic = data["model"]
le_sex = data["le_sex"]

def show_explore_page():
    st.title("Explore Diabetes Data")
   # Membagi data menjadi fitur dan target
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']

    # Membuat prediksi
    y_pred = logistic.predict(X)

    # Menghasilkan confusion matrix
    cm = confusion_matrix(y, y_pred)
    cm_labels = ['No Diabetes', 'Diabetes']

    # Menampilkan confusion matrix menggunakan Plotly
    fig_cm = ff.create_annotated_heatmap(z=cm, x=cm_labels, y=cm_labels, colorscale='Blues', showscale=True, annotation_text=cm.astype(str))
    fig_cm.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')
    
    st.write("Confusion Matrix")
    st.plotly_chart(fig_cm)

show_explore_page()