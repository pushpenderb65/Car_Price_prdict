import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
@st.cache_resource
def load_model():
    return joblib.load('car_price_model.pkl')

pipeline = load_model()

st.title("🚗 Ford Car Price Predictor")
st.write("Enter the details of the Ford car to estimate its price.")

# Form for user input
with st.form("car_details_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # Categorical Inputs
        # (You can expand these lists based on your specific ford.csv unique values)
        model = st.selectbox('Car Model', [' Fiesta', ' Focus', ' Puma', ' Kuga', ' EcoSport', ' C-MAX', ' Mondeo', ' Ka+', ' Tourneo Custom', ' B-MAX', ' S-MAX', ' Edge', ' Tourneo Connect', ' Grand C-MAX', ' KA', ' Galaxy', ' Mustang', ' Grand Tourneo Connect', ' Fusion', ' Ranger', ' Streetka', ' Escort', ' Transit Tourneo'])
        transmission = st.selectbox('Transmission', ['Manual', 'Automatic', 'Semi-Auto'])
        fuelType = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'Hybrid', 'Electric', 'Other'])
        year = st.number_input('Production Year', min_value=1990, max_value=2024, value=2018, step=1)
        
    with col2:
        # Numerical Inputs
        mileage = st.number_input('Mileage', min_value=0, max_value=300000, value=15000, step=500)
        tax = st.number_input('Annual Tax (£)', min_value=0, max_value=1000, value=145, step=5)
        mpg = st.number_input('Miles Per Gallon (MPG)', min_value=10.0, max_value=250.0, value=50.0, step=1.0)
        engineSize = st.number_input('Engine Size (L)', min_value=0.0, max_value=6.0, value=1.0, step=0.1)
        
    submit_button = st.form_submit_button("Predict Price")

# Prediction Logic
if submit_button:
    # Create a DataFrame from the user input
    input_data = pd.DataFrame({
        'model': [model],
        'year': [year],
        'transmission': [transmission],
        'mileage': [mileage],
        'fuelType': [fuelType],
        'tax': [tax],
        'mpg': [mpg],
        'engineSize': [engineSize]
    })
    
    # Predict using the loaded pipeline
    prediction = pipeline.predict(input_data)[0]
    
    st.success(f"### Estimated Price: £{prediction:,.2f}")