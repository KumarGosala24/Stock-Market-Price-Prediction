#step1:
#installed required packages

!pip install yfinance
!pip install pandas numpy matplotlib scikit-learn
!pip install keras tensorflow
!pip install ta

#step2:
#import necessary libarraies

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
import math

# Technical Indicator
import ta
def add_technical_indicators(df):
    df = df.copy()
    close = df['Close']  # ✅ Always use Series, not DataFrame

    # Technical indicators
    df['SMA_20'] = ta.trend.sma_indicator(close, window=20)
    df['EMA_20'] = ta.trend.ema_indicator(close, window=20)
    df['RSI'] = ta.momentum.rsi(close, window=14)
    df['MACD'] = ta.trend.macd(close)

    bb = ta.volatility.BollingerBands(close, window=20)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()

    df.dropna(inplace=True)
    return df

#Step 3:
#Load stock data

tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS','TATASTEEL.NS']
stock_data = {}

for ticker in tickers:
    df = yf.download(ticker, start='2010-01-01', end='2024-12-31')[['Close']]
    df.dropna(inplace=True)
    stock_data[ticker] = df


#step4: Data Visualization
for ticker, df in stock_data.items():
    plt.figure(figsize=(12, 5))
    plt.title(f'{ticker} Closing Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date')
    plt.ylabel('Close Price INR')
    plt.show()


#step5: Data preprocessing

def preprocess_data(df):
    data = df.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    train_len = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_len]
    test_data = scaled_data[train_len - 60:]
    # Get the index of the 'Close' column after scaling
    close_column_index = df.columns.get_loc('Close')
    return scaler, train_data, test_data, train_len, close_column_index


#Step 6.Create sequence for LStM

def create_multi_feature_dataset(dataset, time_step=60, output_feature_index=0):
    x, y = [], []
    for i in range(time_step, len(dataset)):
        x.append(dataset[i-time_step:i])
        y.append(dataset[i, output_feature_index])  # Use the specified output feature index as target
    return np.array(x), np.array(y)


# Step 7. Build and train the LSTM model

models = {}
for ticker, df in stock_data.items():
    print(f"Training model for {ticker}...")
    scaler, train_data, test_data, train_len, close_column_index = preprocess_data(df)
    x_train, y_train = create_multi_feature_dataset(train_data, output_feature_index=close_column_index)
    x_test, y_test = create_multi_feature_dataset(test_data, output_feature_index=close_column_index)

    # Reshape x_train and x_test to be 3D [samples, time_steps, features]
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2])
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2])

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=5, batch_size=64, verbose=1)

    models[ticker] = (model, scaler, x_test, y_test, df, train_len)


# Step 8: Prediction and Performance Evaluation

for ticker, (model, scaler, x_test, y_test, df, train_len) in models.items():
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = math.sqrt(mean_squared_error(y_test_scaled, predictions))
    print(f'{ticker} RMSE: {rmse:.2f}')


#Step 9: Plot the Results

for ticker, (model, scaler, x_test, y_test, df, train_len) in models.items():
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    train = df[:train_len]
    valid = df[train_len:]
    valid = valid.copy()
    valid['Predictions'] = predictions

    plt.figure(figsize=(14,6))
    plt.title(f'{ticker} - Actual vs Predicted Closing Prices')
    plt.plot(train['Close'], label='Training')
    plt.plot(valid['Close'], label='Actual')
    plt.plot(valid['Predictions'], label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.show()

















































































