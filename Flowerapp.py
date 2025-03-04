import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open('flower_classifier.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Species label mapping
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Streamlit UI
st.title("Flower Classification")
st.write("Enter flower dimensions to predict the species.")

# User inputs
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")

# Prediction button
if st.button("Predict Species"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_species = species_map[prediction[0]]
    
    st.success(f'Predicted Species: **{predicted_species}** 🌿')