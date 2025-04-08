
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import ta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

# Step 1: Load the Stock Data
def load_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
#     data['Adj Close'] = data['Adj Close'].squeeze()

    data['Return'] = data['Adj Close'].pct_change()
    data['RSI'] = ta.momentum.RSIIndicator(data['Adj Close'].squeeze()).rsi()
    data['EMA'] = ta.trend.EMAIndicator(data['Adj Close'].squeeze()).ema_indicator()
    data.dropna(inplace=True)
    return data

# Step 2: Prepare the Dataset
def prepare_data(data, lookback=14):
    features = ['Adj Close', 'Volume', 'RSI', 'EMA']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    def create_sequences(data, lookback):
        X_test, y_test = [], []
        for i in range(lookback, len(data)):
            X_test.append(data[i-lookback:i])
            y_test.append(data[i, 0])  # Only the 'Adj Close' column
        return np.array(X_test), np.array(y_test)

    X_test, y_test = create_sequences(scaled_data, lookback)
    test_dates = data.index[lookback:]  # Keep corresponding dates for plotting

    return X_test, y_test, scaler, features, test_dates

# Step 3: Model Loading and Prediction
def load_and_predict(model_file, X_test, scaler, features):
    # Load the trained model
    model = load_model(model_file, custom_objects={'mse': MeanSquaredError()})

    # Make predictions
    y_pred = model.predict(X_test)

    # Rescale predictions
    dummy_features = np.zeros((len(y_pred), len(features) - 1))
    y_pred_rescaled = scaler.inverse_transform(np.concatenate([y_pred, dummy_features], axis=1))[:, 0]

    return y_pred_rescaled

# Step 4: Streamlit Web App
st.title("📈 Stock Price Prediction with LSTM")
st.markdown("This app uses an LSTM model to predict stock prices based on historical data.")

# User Input for Stock
stock_symbol = st.text_input("Enter Stock Ticker", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.date_input("End Date", pd.Timestamp.now().strftime('%Y-%m-%d'))

# Load data and prepare for prediction
data = load_stock_data(stock_symbol, start_date, end_date)
X_test, y_test, scaler, features, test_dates = prepare_data(data)

# Load the model and make predictions
y_pred_rescaled = load_and_predict('lstm_stock_model.keras', X_test, scaler, features)

# **🔹 Fix: Convert `y_test` back to actual prices**
y_test_rescaled = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), len(features) - 1))], axis=1)
)[:, 0]

# 📌 Ensure predictions are 1D for plotting
y_pred_rescaled = y_pred_rescaled.flatten()
y_test_rescaled = y_test_rescaled.flatten()

# **Plot True vs Predicted Prices**
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test_dates, y_test_rescaled, label="Actual Prices", linestyle="solid", alpha=0.7, color="blue")
ax.plot(test_dates, y_pred_rescaled, label="Predicted Prices", linestyle="dashed", color="red", alpha=0.7)
ax.set_title(f"True vs Predicted Prices for {stock_symbol} (2024)")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

# Show Prediction Results
st.write(f"📌 **Predicted Prices for {stock_symbol} (Last 5 Days):**")
predictions_df = pd.DataFrame({"Date": test_dates[-5:], "Predicted Price": y_pred_rescaled[-5:]})
st.write(predictions_df)
