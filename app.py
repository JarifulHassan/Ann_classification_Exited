import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import sys

# Load trained model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open("one_hot_encoder_geo.pkl", 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open("label_encoder_gender.pkl", 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")

# User input
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
credit_score = st.number_input("Credit Score", min_value=0)
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0)
tenure = st.number_input("Tenure", min_value=0)
balance = st.number_input("Balance")
num_of_products = st.number_input("Number of Products", min_value=1)
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary")

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == "Yes" else 0],
    'IsActiveMember': [1 if is_active_member == "Yes" else 0],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode the 'Geography' feature
try:
    # Replace 'Geography' with the actual variable holding the geography value
    geo_encoded =     geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out())


except Exception as e:
    st.error(f"An error occurred: {e}")
    raise e  # Re-raise the caught exception

# Combine input data with one-hot encoded features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

# Display result
if prediction_probability > 0.5:
    st.write('Person is likely to churn')
else:
    st.write('Person is not likely to churn')
