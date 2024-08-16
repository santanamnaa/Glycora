# Glycora

**Glycora** is an AI-powered diabetes prediction app that utilizes personal health data to predict the risk of diabetes. The app is built using **Streamlit** and **machine learning models** to provide real-time predictions and explore diabetes data.

## Features

- **Prediction Page**: Enter personal data to receive a diabetes risk prediction.
- **Explore Page**: Visualize and analyze diabetes data, including a confusion matrix for model performance.

## Demo

https://github.com/user-attachments/assets/d2f7ca7b-1e82-478b-b5d1-65e349174fc1

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/santanamnaa/Glycora.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Glycora
   ```
3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

### Prediction Page:

1. Go to the **Diabetes Prediction** page.
2. Input the required data such as sex, age, BMI, blood glucose level, and other health metrics.
3. Click "Calculate Diabetes" to get the prediction result.

### Explore Page:

1. Go to the **Explore Diabetes Data** page.
2. View a confusion matrix of the model’s predictions and explore the cleaned diabetes dataset.

## Model Information

The model used for diabetes prediction is stored in the `saved_steps.pkl` file, which includes:

- A trained logistic regression model.
- Encoders for transforming categorical variables.
