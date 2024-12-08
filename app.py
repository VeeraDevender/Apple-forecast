# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from joblib import load

# Streamlit App
st.title("Stock Price Prediction App")

st.write("""
## Upload your stock dataset to get started
The dataset should include Open, High, Low, Close prices with 'Date' as a column.
""")

# Upload file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

# Function to create dataset for LSTM model
def create_dataset(dataset, time_step=60):
    X = []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), 0])
    return np.array(X)

# Load pre-trained models
@st.cache_resource
def load_models():
    lstm_model = load_model("lstm_model.h5")
    scaler = load('scaler_model.pkl')
    return lstm_model, scaler

lstm_model, scaler = load_models()

# Main code block
if uploaded_file is not None:
    # Load and display dataset
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(data.head())

    # Preprocessing
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Scaling the 'Close' column for modeling
    scaled_data = scaler.transform(data['Close'].values.reshape(-1, 1))

    # Creating dataset for prediction
    time_step = 60
    last_60_days = scaled_data[-time_step:]
    forecast_input = last_60_days.reshape(1, -1, 1)

    # Forecast next 30 days
    st.write("### 30-Day Forecasted Closing Prices")
    forecasted_prices = []
    for _ in range(30):  # Predicting for the next 30 days
        forecasted_price = lstm_model.predict(forecast_input)[0][0]
        forecasted_prices.append(forecasted_price)
        forecast_input = np.append(forecast_input[:, 1:, :], [[[forecasted_price]]], axis=1)

    # Convert forecasted prices back to the original scale
    forecasted_prices = scaler.inverse_transform(np.array(forecasted_prices).reshape(-1, 1))

    # Plot the forecasted prices along with historical data
    forecast_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30)

    last_30_days = data['Close'][-30:]

    # Plot the historical data and the forecasted prices
    plt.figure(figsize=(14, 7))
    plt.plot(last_30_days.index, last_30_days.values, color='blue', label='Last 30 Days Actual Price')
    plt.plot(forecast_dates, forecasted_prices, color='orange', label='30-Day Forecasted Price')
    plt.title('Last 30 Days Historical Close Price and 30-Day Forecast')
    plt.xlabel('Date')
    plt.ylabel('Stock Price USD ($)')
    plt.legend()
    st.pyplot(plt)

    # Display forecasted prices
    st.write("### Predicted Prices for the Next 30 Days")
    predicted_df = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted Close Price': forecasted_prices.flatten()
    })
    st.write(predicted_df)

else:
    st.write("Please upload a CSV file to proceed.")
