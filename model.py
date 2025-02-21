import json
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load sold stock data
with open('sold_stock1.json') as f:
    sold_stock = json.load(f)

# Create DataFrame from the loaded JSON data
sold_df = pd.DataFrame(sold_stock)

# Preprocess the sold stock data
sold_df['Date'] = pd.to_datetime(sold_df['Date'])
sold_df['Quantity Sold'] = 1  # Assuming each entry in sold stock represents one sale

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

# Function to create sequences for LSTM
def create_sequences(data, past_steps, future_steps):
    X, y = [], []
    for i in range(len(data) - past_steps - future_steps):
        X.append(data[i:i + past_steps])
        y.append(data[i + past_steps:i + past_steps + future_steps])
    return np.array(X), np.array(y)

# Parameters
past_steps = 30  # Number of days to look back
future_steps = 90  # Number of days to predict

# List of unique models
models = daily_sales['Model'].unique()

predictions = {}

# Loop through each model
for model in models:
    model_data = daily_sales[daily_sales['Model'] == model]['Quantity Sold'].values

    # Check if there are enough data points to generate sequences
    if len(model_data) < past_steps + future_steps:
        print(f"Not enough data points for model {model}.")
        continue

    # Create sequences
    X, y = create_sequences(model_data, past_steps, future_steps)

    # Reshape X for LSTM (samples, timesteps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build the LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(past_steps, 1)))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(50, return_sequences=False))
    lstm_model.add(Dense(future_steps))

    # Compile the model
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    lstm_model.fit(X, y, epochs=100, batch_size=1, verbose=2)

    # Predict the next 90 days based on the last 30 days of data for the model
    last_30_days = model_data[-past_steps:]  # Get the last 30 days
    last_30_days = last_30_days.reshape((1, past_steps, 1))  # Reshape for LSTM

    # Make prediction for the upcoming 90 days
    next_90_days_prediction = lstm_model.predict(last_30_days)

    # Store the predictions for this model
    predictions[model] = next_90_days_prediction.flatten()

# Get the list of unique categories
categories = daily_sales['Category'].unique()

# Plot predictions for each category separately
for category in categories:
    plt.figure(figsize=(12, 6))

    # Loop through the predictions for each model within the category
    for model, prediction in predictions.items():
        model_category = daily_sales[daily_sales['Model'] == model]['Category'].iloc[0]
        if model_category == category:
            plt.plot(prediction, label=model)

    plt.title(f'Predicted Quantity Sold for the Next 90 Days - Category {category}')
    plt.xlabel('Days')
    plt.ylabel('Quantity Sold')
    plt.legend()
    plt.grid()
    plt.show()

# Output predictions
for model, prediction in predictions.items():
    print(f"Predicted sales for the next 90 days for model {model}: {prediction}")


# ----------------------------------------------------------------------------------------------------------
# import json
# import numpy as np
# import pandas as pd
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

# # Load sold stock data
# with open('sold_stock1.json') as f:
#     sold_stock = json.load(f)

# # Create DataFrame from the loaded JSON data
# sold_df = pd.DataFrame(sold_stock)

# # Preprocess the sold stock data
# sold_df['Date'] = pd.to_datetime(sold_df['Date'])
# sold_df['Quantity Sold'] = 1  # Assuming each entry in sold stock represents one sale

# # Aggregate daily sales data for each model
# daily_sales = sold_df.groupby(['Category', 'Model', 'Date'])['Quantity Sold'].sum().reset_index()

# # Fill in missing days with zero sales if needed
# all_days = pd.date_range(start=daily_sales['Date'].min(), end=daily_sales['Date'].max(), freq='D')
# daily_sales = (
#     daily_sales.set_index('Date')
#     .groupby(['Category', 'Model'])
#     .apply(lambda x: x.reindex(all_days, fill_value=0))
#     .drop(columns=['Category', 'Model'])
#     .reset_index()
#     .rename(columns={'level_2': 'Date'})
# )

# # Function to create sequences for LSTM
# def create_sequences(data, past_steps, future_steps):
#     X, y = [], []
#     for i in range(len(data) - past_steps - future_steps):
#         X.append(data[i:i + past_steps])
#         y.append(data[i + past_steps:i + past_steps + future_steps])
#     return np.array(X), np.array(y)

# # Parameters
# past_steps = 30  # Number of days to look back
# future_steps = 90  # Number of days to predict

# # List of unique models
# models = daily_sales['Model'].unique()

# predictions = {}

# # Loop through each model
# for model in models:
#     model_data = daily_sales[daily_sales['Model'] == model]['Quantity Sold'].values

#     # Check if there are enough data points to generate sequences
#     if len(model_data) < past_steps + future_steps:
#         print(f"Not enough data points for model {model}.")
#         continue

#     # Create sequences
#     X, y = create_sequences(model_data, past_steps, future_steps)

#     # Reshape X for LSTM (samples, timesteps, features)
#     X = X.reshape((X.shape[0], X.shape[1], 1))

#     # Build the LSTM model
#     lstm_model = Sequential()
#     lstm_model.add(LSTM(50, return_sequences=True, input_shape=(past_steps, 1)))
#     lstm_model.add(Dropout(0.2))
#     lstm_model.add(LSTM(50, return_sequences=False))
#     lstm_model.add(Dense(future_steps))

#     # Compile the model
#     lstm_model.compile(optimizer='adam', loss='mean_squared_error')

#     # Train the model
#     lstm_model.fit(X, y, epochs=100, batch_size=1, verbose=2)

#     # Predict the next 90 days based on the last 30 days of data for the model
#     last_30_days = model_data[-past_steps:]  # Get the last 30 days
#     last_30_days = last_30_days.reshape((1, past_steps, 1))  # Reshape for LSTM

#     # Make prediction for the upcoming 90 days
#     next_90_days_prediction = lstm_model.predict(last_30_days)

#     # Store the predictions for this model
#     predictions[model] = next_90_days_prediction.flatten()

# # Get the list of unique categories
# categories = daily_sales['Category'].unique()

# # Plot predictions for each category separately using bar graphs
# for category in categories:
#     plt.figure(figsize=(12, 6))

#     # Loop through the predictions for each model within the category
#     for model, prediction in predictions.items():
#         model_category = daily_sales[daily_sales['Model'] == model]['Category'].iloc[0]
#         if model_category == category:
#             # Plot using bar graph instead of line graph
#             plt.bar(np.arange(1, future_steps + 1), prediction, width=0.2, label=model, align='center')

#     plt.title(f'Predicted Quantity Sold for the Next 90 Days - Category {category}')
#     plt.xlabel('Days')
#     plt.ylabel('Quantity Sold')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# # Output predictions
# for model, prediction in predictions.items():
#     print(f"Predicted sales for the next 90 days for model {model}: {prediction}")
