
import streamlit as st
import pandas as pd
import joblib

# Load the trained model
try:
    model = joblib.load('logi.sav')
except FileNotFoundError:
    st.error("Model file 'logi.sav' not found. Please ensure it's in the same directory.")
    st.stop()

# Define the feature names (must match the order used during training)
feature_names = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
                 'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
                 'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
                 'Warehouse_Processing_Time']

# Streamlit App Title
st.title('Delivery Delay Prediction App')

st.write("Enter the feature values below to predict if there will be a delivery delay (1 for delay, 0 for no delay).")

# Create input widgets for each feature
input_data = {}
for feature in feature_names:
    # Using st.number_input for numerical features. You might want to adjust min_value, max_value, and step for specific features.
    input_data[feature] = st.number_input(f'Enter {feature.replace("_", " ")}', value=0.0, step=0.1)

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Make prediction when button is clicked
if st.button('Predict Delivery Delay'):
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        st.subheader('Prediction Result:')
        if prediction[0] == 1:
            st.success("**Delivery Delay: YES**")
        else:
            st.info("**Delivery Delay: NO**")

        st.write(f"Probability of No Delay: {prediction_proba[0][0]:.4f}")
        st.write(f"Probability of Delay: {prediction_proba[0][1]:.4f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
