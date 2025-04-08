# import yfinance as yf
# import ta
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
# from tensorflow.keras.optimizers import Adam
# import tensorflow as tf

# # Load stock data
# def load_stock_data(ticker, start_date, end_date):
#     data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
#     data.reset_index(inplace=True)
#     # Ensure 'Adj Close' is a 1D Series
#     adj_close = data['Adj Close'].squeeze()
#     data['Return'] = adj_close.pct_change()

#     # Compute Technical Indicators
#     data['RSI'] = ta.momentum.RSIIndicator(adj_close).rsi()
#     data['EMA'] = ta.trend.EMAIndicator(adj_close).ema_indicator()
#     data['ATR'] = ta.volatility.AverageTrueRange(
#         high=data['High'].squeeze(), 
#         low=data['Low'].squeeze(), 
#         close=adj_close
#     ).average_true_range()
    
#     data['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
#         high=data['High'].squeeze(), 
#         low=data['Low'].squeeze(), 
#         close=adj_close, 
#         volume=data['Volume'].squeeze()
#     ).volume_weighted_average_price()

#     data.dropna(inplace=True)
#     return data

# # Define parameters
# stock_symbol = "GOOGL"
# start_date = "2018-01-01"
# end_date = "2023-12-31"
# data = load_stock_data(stock_symbol, start_date, end_date)
# features = ['Adj Close', 'Volume', 'RSI', 'EMA', 'VWAP', 'ATR']
# lookback = 7

# # Scale data
# scaler = MinMaxScaler()
# scaled_data = scaler.fit_transform(data[features])

# def create_sequences(data, lookback):
#     X, y = [], []
#     for i in range(lookback, len(data)):
#         X.append(data[i-lookback:i])
#         y.append(data[i, 0])
#     return np.array(X), np.array(y)

# X, y = create_sequences(scaled_data, lookback)
# train_size = int(0.8 * len(X))
# X_train, X_test = X[:train_size], X[train_size:]
# y_train, y_test = y[:train_size], y[train_size:]

# # Define Generator
# def build_generator():
#     model = Sequential([
#         LSTM(128, activation='relu', return_sequences=True, input_shape=(lookback, len(features))),
#         Dropout(0.3),
#         LSTM(64, activation='relu', return_sequences=False),
#         Dense(len(features))
#     ])
#     return model

# # Define Discriminator
# def build_discriminator():
#     model = Sequential([
#         LSTM(64, activation='relu', return_sequences=True, input_shape=(lookback, len(features))),
#         Dropout(0.3),
#         LSTM(32, activation='relu', return_sequences=False),
#         Dense(1, activation='sigmoid')
#     ])
#     return model

# # Build GAN
# generator = build_generator()
# discriminator = build_discriminator()
# discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

# discriminator.trainable = False
# z = Input(shape=(lookback, len(features)))
# generated_data = generator(z)
# validity = discriminator(generated_data)
# gan = Model(z, validity)
# gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# # Training GAN
# def train_gan(epochs=50, batch_size=32):
#     for epoch in range(epochs):
#         idx = np.random.randint(0, X_train.shape[0], batch_size)
#         real_data = X_train[idx]
#         fake_data = generator.predict(real_data)
#         valid = np.ones((batch_size, 1))
#         fake = np.zeros((batch_size, 1))
        
#         d_loss_real = discriminator.train_on_batch(real_data, valid)
#         d_loss_fake = discriminator.train_on_batch(fake_data, fake)
#         d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
#         g_loss = gan.train_on_batch(real_data, valid)
        
#         if epoch % 50 == 0:
#             print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# train_gan()

# # Generate Predictions
# def generate_predictions():
#     test_input = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
#     predicted_scaled = generator.predict(test_input)
#     dummy_features = np.zeros((predicted_scaled.shape[0], len(features) - 1))
#     predicted_price = scaler.inverse_transform(np.concatenate([predicted_scaled, dummy_features], axis=1))[:, 0]
#     return predicted_price

# tomorrow_price = generate_predictions()[0]
# today_price = float(data['Adj Close'].iloc[-1])

# print(f"Tomorrow's Predicted Price: {tomorrow_price:.2f}, Today: {today_price:.2f}")


import yfinance as yf
import ta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Load stock data
def load_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    data.reset_index(inplace=True)
    
    adj_close = data['Adj Close'].squeeze()
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

    
    data.dropna(inplace=True)
    return data

# Parameters
stock_symbol = "GOOGL"
start_date = "2018-01-01"
end_date = "2023-12-31"
data = load_stock_data(stock_symbol, start_date, end_date)
features = ['Adj Close', 'Volume', 'RSI', 'EMA', 'VWAP', 'ATR']
lookback = 7

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

# Function to create sequences
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])  
        y.append(data[i, 0])  
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, lookback)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Generator Model
def build_generator():
    model = Sequential([
        LSTM(128, activation='relu', return_sequences=True, input_shape=(lookback, len(features))),
        Dropout(0.3),
        LSTM(64, activation='relu', return_sequences=True),  # Ensure output is a sequence
        LSTM(32, activation='relu', return_sequences=False),
        Dense(len(features))
    ])
    return model

# Discriminator Model
def build_discriminator():
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, input_shape=(lookback, len(features))),
        Dropout(0.3),
        LSTM(32, activation='relu', return_sequences=False),
        Dense(1, activation='sigmoid')
    ])
    return model

# Build Models
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False
z = Input(shape=(lookback, len(features)))
generated_data = generator(z)
validity = discriminator(generated_data)
gan = Model(z, validity)
gan.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

# Train GAN
def train_gan(epochs=100, batch_size=32):
    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_data = X_train[idx]

        fake_data = generator.predict(real_data)  # Ensure generator produces sequences
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        d_loss_real = discriminator.train_on_batch(real_data, valid)
        d_loss_fake = discriminator.train_on_batch(fake_data, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        g_loss = gan.train_on_batch(real_data, valid)

        if epoch % 10 == 0:
            print(f"Epoch {epoch} - D Loss: {d_loss[0]:.4f}, G Loss: {g_loss:.4f}")

train_gan()

# Generate Predictions
def generate_predictions():
    test_input = X_test[-1].reshape(1, X_test.shape[1], X_test.shape[2])
    predicted_scaled = generator.predict(test_input)

    dummy_features = np.zeros((predicted_scaled.shape[0], len(features) - 1))
    predicted_full = np.concatenate([predicted_scaled, dummy_features], axis=1)
    
    predicted_price = scaler.inverse_transform(predicted_full)[:, 0]
    return predicted_price

tomorrow_price = generate_predictions()[0]
today_price = float(data['Adj Close'].iloc[-1])

print(f"Tomorrow's Predicted Price: {tomorrow_price:.2f}, Today: {today_price:.2f}")


