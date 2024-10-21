"""
Laboratorio 08
@autors:
    - José Pablo Kiesling Lange, 21581
    - Melissa Pérez Alarcón, 21385

Interfaz de usuario para predecir el alquiler de propiedades
"""
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

with open('src/random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('src/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('src/model_columns.pkl', 'rb') as file:
    model_columns = pickle.load(file)

data_path = 'data/houses_to_rent_v2.csv'
data = pd.read_csv(data_path)

st.title('Predicción de Alquiler de Viviendas en Brasil')
st.write('Ingrese los detalles de la propiedad para predecir el alquiler mensual.')

area = st.number_input('Área (en m²)', min_value=0.0, step=1.0)
rooms = st.number_input('Número de habitaciones', min_value=0, step=1)
bathroom = st.number_input('Número de baños', min_value=0, step=1)
parking_spaces = st.number_input('Número de estacionamientos', min_value=0, step=1)
hoa = st.number_input('HOA (R$)', min_value=0.0, step=1.0)
property_tax = st.number_input('Impuesto a la propiedad (R$)', min_value=0.0, step=1.0)
fire_insurance = st.number_input('Seguro contra incendios (R$)', min_value=0.0, step=1.0)
floor_input = st.text_input('Piso', value='-')

city = st.selectbox('Ciudad', data['city'].unique())
animal = st.selectbox('¿Se permiten animales?', data['animal'].unique())
furniture = st.selectbox('¿Está amueblado?', data['furniture'].unique())

if st.button('Predecir'):
    try:
        input_dict = {
            'area': area,
            'rooms': rooms,
            'bathroom': bathroom,
            'parking spaces': parking_spaces,
            'hoa (R$)': hoa,
            'property tax (R$)': property_tax,
            'fire insurance (R$)': fire_insurance,
            'floor': floor_input,
            'city': city,
            'animal': animal,
            'furniture': furniture
        }

        input_df = pd.DataFrame([input_dict])

        categorical_columns = ['city', 'animal', 'furniture', 'floor']
        numeric_columns = ['area', 'rooms', 'bathroom', 'parking spaces', 'hoa (R$)', 'property tax (R$)', 'fire insurance (R$)']

        input_df_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)
        input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)
        input_df_encoded[numeric_columns] = scaler.transform(input_df_encoded[numeric_columns])

        prediction = model.predict(input_df_encoded)

        st.success(f'El alquiler mensual estimado es: R$ {prediction[0]:.2f}')

    except Exception as e:
        st.error(f'Ocurrió un error durante la predicción: {e}')

st.subheader('Importancia de las características en el modelo')
feature_importances = model.feature_importances_
feature_importances_df = pd.DataFrame({
    'Característica': model_columns,
    'Importancia': feature_importances
}).sort_values(by='Importancia', ascending=False)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='Importancia', y='Característica', data=feature_importances_df.head(10), ax=ax)
st.pyplot(fig)

st.subheader('Tendencias de alquiler por ciudad')
avg_rent_by_city = data.groupby('city')['rent amount (R$)'].mean().reset_index()

fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.barplot(x='rent amount (R$)', y='city', data=avg_rent_by_city, ax=ax2)
ax2.set_xlabel('Alquiler promedio (R$)')
ax2.set_ylabel('Ciudad')
st.pyplot(fig2)
