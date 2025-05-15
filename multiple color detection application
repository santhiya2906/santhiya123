import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 1. Data Collection
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# 2. Data Cleaning
def clean_data(df):
    df = df.dropna()  # Handle missing values
    df = df.drop_duplicates()  # Remove duplicates
    df.index = pd.to_datetime(df.index)  # Standardize date format
    return df

# 3. Exploratory Data Analysis
def perform_eda(df, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label=f'{ticker} Close Price')
    plt.title(f'{ticker} Stock Price Trend')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('stock_trend.png')
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation')
    plt.savefig('correlation_heatmap.png')
    plt.close()

# 4. Feature Engineering
def engineer_features(df):
    df['MA50'] = df['Close'].rolling(window=50).mean()  # 50-day moving averageいる

    df['Lag1'] = df['Close'].shift(1)  # 1-day lag
    return df

# 5. ARIMA Model
def train_arima_model(data, order=(5,1,0)):
    model = ARIMA(data, order=order)
    fitted_model = model.fit()
    return fitted_model

# 6. LSTM Model
def prepare_lstm_data(data, look_back=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

def train_lstm_model(X_train, y_train, epochs=50, batch_size=32):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

# 7. Model Evaluation
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape

# Main execution
def main():
    # Parameters
    ticker = 'AAPL'  # Example: Apple stock
    start_date = '2020-01-01'
    end_date = '2025-04-30'
    train_split = 0.8
    look_back = 60

    # Fetch and clean data
    df = fetch_stock_data(ticker, start_date, end_date)
    df = clean_data(df)
    
    # Perform EDA
    perform_eda(df, ticker)
    
    # Feature engineering
    df = engineer_features(df)
    
    # Prepare data for models
    close_prices = df['Close'].values
    train_size = int(len(close_prices) * train_split)
    train_data = close_prices[:train_size]
    test_data = close_prices[train_size:]
    
    # ARIMA Model
    arima_model = train_arima_model(train_data)
    arima_predictions = arima_model.forecast(steps=len(test_data))
    
    # LSTM Model
    X, y, scaler = prepare_lstm_data(close_prices, look_back)
    X_train = X[:int(len(X) * train_split)]
    y_train = y[:int(len(y) * train_split)]
    X_test = X[int(len(X) * train_split):]
    y_test = y[int(len(y) * train_split):]
    
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    lstm_model = train_lstm_model(X_train, y_train)
    lstm_predictions = lstm_model.predict(X_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)
    y_test = scaler.inverse_transform([y_test])
    
    # Evaluate models
    arima_rmse, arima_mae, arima_mape = evaluate_model(test_data, arima_predictions)
    lstm_rmse, lstm_mae, lstm_mape = evaluate_model(y_test[0], lstm_predictions.flatten())
    
    # Print evaluation metrics
    print("ARIMA Model Metrics:")
    print(f"RMSE: {arima_rmse:.2f}, MAE: {arima_mae:.2f}, MAPE: {arima_mape:.2f}%")
    print("\nLSTM Model Metrics:")
    print(f"RMSE: {lstm_rmse:.2f}, MAE: {lstm_mae:.2f}, MAPE: {lstm_mape:.2f}%")
    
    # Visualize predictions
    plt.figure(figsize=(12, 6))
    plt.plot(test_data, label='Actual')
    plt.plot(arima_predictions, label='ARIMA Predictions')
    plt.plot(lstm_predictions, label='LSTM Predictions')
    plt.title(f'{ticker} Stock Price Predictions')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig('predictions.png')
    plt.close()

if __name__ == "__main__":
    main()
