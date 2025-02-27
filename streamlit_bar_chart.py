import streamlit as st
import json
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os
import joblib  # To save and load model

# Function to load and preprocess the initial stock data
def load_initial_stock(file):
    initial_stock = json.load(file)
    return pd.DataFrame(initial_stock)

# Function to load and preprocess the sold stock data
def load_sold_stock(file):
    sold_stock = json.load(file)
    return pd.DataFrame(sold_stock)

# Function to preprocess the sold stock data
def preprocess_sold_stock(sold_df):
    sold_df['Date'] = pd.to_datetime(sold_df['Date'])
    sold_df['Quantity Sold'] = 1  # Assuming each entry represents one sale

    # Aggregate daily sales data for each model
    daily_sales = sold_df.groupby(['Category', 'Model', 'Date'])['Quantity Sold'].sum().reset_index()

    # Fill in missing days with zero sales if needed
    all_days = pd.date_range(start=daily_sales['Date'].min(), end=daily_sales['Date'].max(), freq='D')
    daily_sales = (
        daily_sales.set_index('Date')
        .groupby(['Category', 'Model'])
        .apply(lambda x: x.reindex(all_days, fill_value=0))
        .drop(columns=['Category', 'Model'])
        .reset_index()
        .rename(columns={'level_2': 'Date'})
    )
    return daily_sales

# Function to create sequences for LSTM
def create_sequences(data, past_steps, future_steps):
    X, y = [], []
    for i in range(len(data) - past_steps - future_steps):
        X.append(data[i:i + past_steps])
        y.append(data[i + past_steps:i + past_steps + future_steps])
    return np.array(X), np.array(y)

# Function to build and compile the LSTM model
def build_lstm_model(past_steps, future_steps):
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(past_steps, 1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(50, return_sequences=False))
    lstm_model.add(Dense(future_steps, activation='relu'))  # Ensuring non-negative predictions
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    return lstm_model

# Streamlit UI for file upload
st.title("Sales Prediction with LSTM")
st.write("Upload your `initial_stock.json` and `sold_stock.json` files to predict future sales quantities.")

# File Uploads for initial stock and sold stock
initial_stock_file = st.file_uploader("Upload Initial Stock JSON File", type="json")
sold_stock_file = st.file_uploader("Upload Sold Stock JSON File", type="json")

# Check if both files are uploaded
if initial_stock_file is not None and sold_stock_file is not None:
    # Load and preprocess initial stock and sold stock data
    initial_stock_df = load_initial_stock(initial_stock_file)
    sold_stock_df = load_sold_stock(sold_stock_file)

    # Process sold stock data
    daily_sales = preprocess_sold_stock(sold_stock_df)

    # Parameters for the model
    past_steps = 30  # Number of days to look back
    future_steps = 90  # Number of days to predict

    # List of unique models
    models = daily_sales['Model'].unique()

    predictions = {}

    # Directory to save model
    model_directory = "models"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # Loop through each model to train and predict
    for model in models:
        model_data = daily_sales[daily_sales['Model'] == model]['Quantity Sold'].values

        # Check if there are enough data points to generate sequences
        if len(model_data) < past_steps + future_steps:
            st.write(f"Not enough data points for model {model}.")
            continue

        # Create sequences
        X, y = create_sequences(model_data, past_steps, future_steps)

        # Reshape X for LSTM (samples, timesteps, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        # Check if the model has already been trained and saved
        model_filename = os.path.join(model_directory, f"{model}_lstm_model.pkl")
        if os.path.exists(model_filename):
            # If model exists, load it
            lstm_model = joblib.load(model_filename)
            st.write(f"Loaded pre-trained model for {model}.")
        else:
            # If model does not exist, build, train and save it
            lstm_model = build_lstm_model(past_steps, future_steps)
            lstm_model.fit(X, y, epochs=100, batch_size=1, verbose=2)
            joblib.dump(lstm_model, model_filename)  # Save the trained model
            st.write(f"Trained and saved model for {model}.")

        # Predict the next 90 days based on the last 30 days of data for the model
        last_30_days = model_data[-past_steps:]  # Get the last 30 days
        last_30_days = last_30_days.reshape((1, past_steps, 1))  # Reshape for LSTM

        # Make prediction for the upcoming 90 days
        next_90_days_prediction = lstm_model.predict(last_30_days)
        
        # Ensure predictions are non-negative
        next_90_days_prediction = np.clip(next_90_days_prediction, 0, None)

        # Store the predictions for this model
        predictions[model] = next_90_days_prediction.flatten()

    # Get the list of unique categories
    categories = daily_sales['Category'].unique()

    # Plot bar chart predictions for each category
    for category in categories:
        plt.figure(figsize=(12, 6))

        # Loop through the predictions for each model within the category
        for model, prediction in predictions.items():
            model_category = daily_sales[daily_sales['Model'] == model]['Category'].iloc[0]
            if model_category == category:
                days = np.arange(1, future_steps + 1)  # Days from 1 to 90
                plt.bar(days, prediction, label=f'Model {model}', alpha=0.7)

        plt.title(f'Predicted Quantity Sold for the Next 90 Days - Category {category}')
        plt.xlabel('Days')
        plt.ylabel('Quantity Sold')
        plt.legend()
        plt.grid(axis='y')
        st.pyplot(plt)

    # Output predictions
    for model, prediction in predictions.items():
        st.write(f"Predicted sales for the next 90 days for model {model}: {prediction}")
