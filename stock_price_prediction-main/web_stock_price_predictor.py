import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from datetime import datetime

def load_and_predict_stock(stock_symbol):
    # Step 2: Load Stock Data
    end = datetime.now()
    start = datetime(end.year-20, end.month, end.day)  # Fetch 20 years of historical data
    
    # Fetch stock data using yfinance
    stock_data = yf.download(stock_symbol, start=start, end=end)
    
    # Step 3: Data Preprocessing
    # Use the 'Close' price for prediction
    data = stock_data[['Close']]
    
    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data)
    
    # Create sequences of data
    sequence_length = 60  # Use past 60 days to predict the next day's price
    X = []
    y = []
    
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])  # 60 previous days
        y.append(scaled_data[i, 0])  # The next day's price
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshaping X to fit the LSTM model: (samples, time_steps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Step 4: Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    
    # Step 5: Build the LSTM Model
    model = Sequential()
    
    # Add LSTM layers
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))  # Dropout to avoid overfitting
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Add the Dense layer
    model.add(Dense(units=1))  # Output layer with one unit (next day's stock price)
    
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Step 6: Train the Model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    
    # Step 7: Evaluate the Model on Test Data
    predicted_prices = model.predict(X_test)
    
    # Inverse transform the predictions and actual values to get original scale
    predicted_prices = scaler.inverse_transform(predicted_prices)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    return y_test_actual, predicted_prices, stock_data

def main():
    # Streamlit app title
    st.title('Stock Price Prediction')
    
    # Stock symbol input
    stock_symbol = st.text_input('Enter Stock Symbol (e.g., GOOG)', value='GOOG')
    
    # Predict button
    if st.button('Predict Stock Prices'):
        try:
            # Load and predict stock prices
            actual_prices, predicted_prices, stock_data = load_and_predict_stock(stock_symbol)
            
            # Create a Streamlit figure
            fig, ax = plt.subplots(figsize=(10,6))
            ax.plot(actual_prices, color='blue', label=f'Actual {stock_symbol} Price')
            ax.plot(predicted_prices, color='red', label=f'Predicted {stock_symbol} Price')
            ax.set_title(f'{stock_symbol} Stock Price Prediction')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.legend()
            
            # Display the plot in Streamlit
            st.pyplot(fig)
            
            # Display some additional information
            st.write(f"Total number of data points: {len(stock_data)}")
            st.write("Last 5 actual prices:", actual_prices[-5:])
            st.write("Last 5 predicted prices:", predicted_prices[-5:])
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()