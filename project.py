import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('temperature_data.csv')

# Convert 'to_timestamp' to datetime format
df['to_timestamp'] = pd.to_datetime(df['to_timestamp'])

# Define the date range
start_date = pd.to_datetime('2023-01-01')
end_date = pd.to_datetime('2023-12-31')

# Filter the DataFrame for the given date range
df = df[(df['to_timestamp'] >= start_date) & (df['to_timestamp'] <= end_date)]

if 'agg_temp' in df.columns:
    df['agg_temp'] = df['agg_temp'].apply(lambda x: x / 10 if x == 932 else x)

# Identify columns whose names end with "temp"
temp_columns = [col for col in df.columns if col.endswith('temp')]

# Include the to_timestamp column
selected_columns = ['to_timestamp'] + temp_columns

# Extract those columns into a new DataFrame
temp_df = df[selected_columns]

# Remove duplicates
temp_df.drop_duplicates(inplace=True)

# Ensure the data is sorted by timestamp
temp_df.sort_values('to_timestamp', inplace=True)

# Fill missing values (forward fill method)
temp_df.fillna(method='ffill', inplace=True)

# Create lagged features for the temperature columns
for col in temp_columns:
    for lag in range(1, 7):  # Using 6 lags as an example
        temp_df[f'{col}_lag_{lag}'] = temp_df[col].shift(lag)

# Drop rows with NaN values that result from the lagging
temp_df.dropna(inplace=True)

# Initialize a dictionary to store models and predictions
models = {}
predictions = {}

# Iterate over each temperature column to train a model and make predictions
for temp_col in temp_columns:
    print(f'Processing {temp_col}...')
    
    # Define features and target for the current temperature column
    features = [col for col in temp_df.columns if 'lag' in col and col.startswith(temp_col)]
    target = temp_col

    # Split the data into training and testing sets
    X = temp_df[features]
    y = temp_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error for {temp_col}: {mse}')

    # Store the model in the dictionary
    models[temp_col] = model

    # Use the last known values to make a prediction for the next ten minutes
    latest_data = temp_df.tail(1)

    # Prepare the lagged features for the new data point
    new_data = latest_data.copy()
    for lag in range(1, 7):
        new_data[f'{temp_col}_lag_{lag}'] = temp_df[temp_col].shift(lag).tail(1).values[0]

    # Ensure there are no NaN values in the new_data
    new_data.fillna(method='ffill', inplace=True)

    # Predict the temperature for the next ten minutes
    next_temp_pred = model.predict(new_data[features])
    predictions[temp_col] = next_temp_pred[0]
    print(f'Predicted {temp_col} for the next ten minutes: {next_temp_pred[0]}')

# Display all predictions
print("Predictions for the next ten minutes for each temperature column:")
for temp_col, pred in predictions.items():
    print(f'{temp_col}: {pred}')
