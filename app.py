import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pandas as pd
import pickle

# Load the trained model and encoders/scaler
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('Scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('onehotencode.pkl', 'rb') as file:
    ohe_encoder = pickle.load(file)

st.title("Customer Churn Prediction")

# User input
geography = st.selectbox('Geography', ohe_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_credit_card = st.selectbox('Has a credit card', [0, 1])
is_active_member = st.selectbox('Is active member', [0, 1])

# Prepare input data
input_data = {
    'CreditScore': [credit_score],
    'Gender': label_encoder.transform([gender]),  # Encode gender
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_credit_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

input_df = pd.DataFrame(input_data)

# Geography one-hot encoding
geo_encoded = ohe_encoder.transform([[geography]])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_encoder.get_feature_names_out(['Geography']))

# Concatenate all features
final_input = pd.concat([input_df.reset_index(drop=True), geo_encoded_df], axis=1)

# Feature scaling
final_input_scaled = scaler.transform(final_input)

# Make prediction
prediction = model.predict(final_input_scaled)

if prediction > 0.5:
    st.write("THE CUSTOMER IS LIKELY TO CHURN")
else:
    st.write("THE CUSTOMER WON'T CHURN")
