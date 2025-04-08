

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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



# Step 1: Load the Stock Data
def load_stock_data(ticker, start_date, end_date):
    # Fetch stock data
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    
    # Reset index for proper datetime handling
    data.reset_index(inplace=True)
    
    # Ensure 'Adj Close' is a 1D Series
    adj_close = data['Adj Close'].squeeze()
    
    # Compute percentage return
    data['Return'] = adj_close.pct_change()

    # Compute Technical Indicators
    data['RSI'] = ta.momentum.RSIIndicator(adj_close).rsi()
    data['EMA'] = ta.trend.EMAIndicator(adj_close).ema_indicator()
    data['ATR'] = ta.volatility.AverageTrueRange(
        high=data['High'].squeeze(), 
        low=data['Low'].squeeze(), 
        close=adj_close
    ).average_true_range()
    
    data['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
        high=data['High'].squeeze(), 
        low=data['Low'].squeeze(), 
        close=adj_close, 
        volume=data['Volume'].squeeze()
    ).volume_weighted_average_price()

    # Drop NaN values that appear due to indicator calculations
    data.dropna(inplace=True)

    return data

# Step 2: Prepare the Dataset
def prepare_data(data, lookback=7):
    features = ['Adj Close', 'Volume', 'RSI', 'EMA','ATR','VWAP']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    def create_sequences(data, lookback):
        X_test, y_test = [], []
        for i in range(lookback, len(data)):
            X_test.append(data[i-lookback:i])
            y_test.append(data[i, 0])  # 'Adj Close'
        return np.array(X_test), np.array(y_test)

    X_test, y_test = create_sequences(scaled_data, lookback)
    test_dates = data.index[lookback:]  # Corresponding dates

    return X_test, y_test, scaler, features, test_dates

# Step 3: Model Loading and Prediction
def load_and_predict(model_file, X_test, scaler, features):
    model = load_model(model_file, custom_objects={'mse': MeanSquaredError()})
    y_pred = model.predict(X_test)

    # Inverse transform predictions
    dummy_features = np.zeros((len(y_pred), len(features) - 1))
    y_pred_rescaled = scaler.inverse_transform(np.concatenate([y_pred, dummy_features], axis=1))[:, 0]

    return y_pred_rescaled, model

# Step 4: Streamlit Web App
st.title("📈 Stock Price Prediction with LSTM")
st.markdown("This app uses an LSTM model to predict stock prices based on historical data.")

# User Input for Stock
stock_symbol = st.text_input("Enter Stock Ticker", "GOOGL")
start_date = st.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-08-15"))

# Load data
data = load_stock_data(stock_symbol, start_date, end_date)
X_test, y_test, scaler, features, test_dates = prepare_data(data)

# Load model & make predictions
y_pred_rescaled, model = load_and_predict('FeaturesAddedGoogleAttentionModel.keras', X_test, scaler, features)

# Convert y_test back to actual prices
y_test_rescaled = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), len(features) - 1))], axis=1)
)[:, 0]

# **Predict Tomorrow's Price**
latest_data = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])  # Latest sequence
predicted_scaled = model.predict(latest_data)

dummy_features = np.zeros((predicted_scaled.shape[0], len(features) - 1))
predicted_price = scaler.inverse_transform(np.concatenate([predicted_scaled, dummy_features], axis=1))[:, 0]

today_price = float(data['Adj Close'].iloc[-1])
# today_price= predicted_price[1]
tomorrow_price = predicted_price[0]

# Display Tomorrow's Prediction
st.write(f"📈 **Tomorrow's Predicted Price: {tomorrow_price:.2f}** (Today: {today_price:.2f})")

# Trading Decision
if tomorrow_price > today_price:
    st.success(f"✅ **BUY Signal:** Expected Price Increase from {today_price:.2f} to {tomorrow_price:.2f}")
else:
    st.warning(f"❌ **NO BUY:** Expected Price Drop from {today_price:.2f} to {tomorrow_price:.2f}")

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


# **Evaluate Model Accuracy**
mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

# Directional Accuracy
direction_correct = np.sum(np.sign(y_test_rescaled[1:] - y_test_rescaled[:-1]) == np.sign(y_pred_rescaled[1:] - y_pred_rescaled[:-1]))
direction_accuracy = direction_correct / len(y_test_rescaled[1:]) * 100

# Display Metrics
st.write("### Model Evaluation Metrics")
st.write(f"📉 **MSE:** {mse:.4f}")
st.write(f"📉 **RMSE:** {rmse:.4f}")
st.write(f"📉 **MAE:** {mae:.4f}")
st.write(f"📊 **R² Score:** {r2:.4f}")
st.write(f"📈 **Directional Accuracy:** {direction_accuracy:.2f}%")