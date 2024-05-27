import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model
try:
    with open('gold_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'gold_price_model.pkl' is in the correct directory.")
    st.stop()

# Define the feature columns
feature_columns = ['SPX', 'USO', 'SLV', 'EUR/USD']

# Create a Streamlit app
st.title('Gold Price Prediction')

st.write('This app predicts the price of gold based on various market factors.')

# Create input fields for each feature
def get_user_input():
    spx = st.number_input('S&P 500 Index (SPX)', min_value=0.0, format="%.2f")
    uso = st.number_input('United States Oil Fund (USO)', min_value=0.0, format="%.2f")
    slv = st.number_input('iShares Silver Trust (SLV)', min_value=0.0, format="%.2f")
    eur_usd = st.number_input('Euro to US Dollar (EUR/USD)', min_value=0.0, format="%.4f")

    data = {
        'SPX': spx,
        'USO': uso,
        'SLV': slv,
        'EUR/USD': eur_usd
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
user_input = get_user_input()

# Display user input
st.subheader('User Input:')
st.write(user_input)

# Make prediction
if st.button('Predict'):
    try:
        prediction = model.predict(user_input)
        st.subheader('Predicted Gold Price:')
        st.write(prediction[0])
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Running the app
if __name__ == '__main__':
    # Streamlit apps are run using the command line.
    # To run this app, use the command:
    # streamlit run your_script.py
    st.write("Calculate your gold pirce for future investment")
