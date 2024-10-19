"""
Laboratorio 08
@autors:
    - José Pablo Kiesling Lange, 21581
    - Melissa Pérez Alarcón, 21385

Interfaz de usuario para predecir el alquiler de propiedades
"""
import streamlit as st
import pickle

# cargar el modelo
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Predicción de Alquiler')
st.write('Ingrese los detalles de la propiedad para predecir el alquiler mensual')

# campos de entrada para las características
area = st.number_input('Área (en m2)', min_value=0)
rooms = st.number_input('Número de habitaciones', min_value=0)
bathroom = st.number_input('Número de baños', min_value=0)
parking_spaces = st.number_input('Número de estacionamientos', min_value=0)

# el modelo espera 48 inputs, entonces hay que completarlos

# botón para realizar la predicción
if st.button('Predecir'):
    # realiza la predicción
    features = [[area, rooms, bathroom, parking_spaces]]
    prediction = model.predict(features)
    st.write(f'El alquiler mensual estimado es: R$ {prediction[0]}')
