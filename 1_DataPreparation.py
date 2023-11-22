# 1_DataPreparation.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

# Configure logging
logging.basicConfig(filename='ai_scheduling.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def load_data(file_path):
    """
    Loads data from the given file path.

    :param file_path: Path to the data file.
    :return: Loaded DataFrame or None if an error occurs.
    """
    try:
        data = pd.read_csv(file_path)
        logging.info("Data loaded successfully")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def preprocess_data(data):
    """
    Performs data cleaning and preprocessing on the given DataFrame.

    :param data: DataFrame to preprocess.
    :return: Preprocessed DataFrame.
    """
    # Fill missing values
    data.fillna(method='ffill', inplace=True)

    # Convert time columns to datetime format
    time_cols = ['scheduled_departure_time', 'scheduled_arrival_time', 'departure_time', 'arrival_time']
    for col in time_cols:
        data[col] = pd.to_datetime(data[col])

    # Extract hours from time columns
    for col in time_cols:
        data[f'{col}_hour'] = data[col].dt.hour

    return data

def handle_uncertainties(data):
    """
    Adjusts the dataset to handle uncertainties such as variable flight times 
    and rapidly changing conditions.

    :param data: DataFrame containing the schedule data.
    :return: Adjusted DataFrame.
    """
    # Define a mapping of weather conditions to delay factor
    weather_delay_factor = {
        'clear': 1.0, 'rainy': 1.1, 'stormy': 1.2, 'foggy': 1.15
    }

    # Apply delay factor to flight duration
    data['adjusted_flight_duration'] = data.apply(
        lambda row: row['flight_duration'] * weather_delay_factor.get(row['weather_condition'], 1),
        axis=1
    )

    # Adjust flight status based on crew availability
    data['flight_status'] = data.apply(
        lambda row: 'delayed' if row['crew_on_standby'] < 3 and row['weather_condition'] == 'stormy' else 'on_time',
        axis=1
    )

    return data


def encode_and_scale(data, categorical_columns, numerical_columns):
    """
    Encodes categorical variables and scales numerical columns.

    :param data: DataFrame containing the data.
    :param categorical_columns: List of names of categorical columns.
    :param numerical_columns: List of names of numerical columns.
    :return: DataFrame with encoded and scaled data.
    """
  # Encoding categorical columns
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])

    # Scaling numerical columns
    scaler = MinMaxScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    return data


# Main execution
if __name__ == "__main__":
    try:
        # Load and preprocess data
        data = load_data('your_dataset.csv')
        if data is None:
            raise ValueError("Data file not found or invalid.")

        data = preprocess_data(data)
        data = handle_uncertainties(data)

        # Encoding and scaling
        categorical_cols = ['weather_condition', 'aircraft_type', 'destination_airport', 'departure_airport']
        numerical_cols = ['scheduled_departure_time_hour', 'scheduled_arrival_time_hour', 'departure_time_hour', 'arrival_time_hour', 'crew_on_standby', 'flight_duration', 'adjusted_flight_duration']
        data = encode_and_scale(data, categorical_cols, numerical_cols)

        # Splitting data
        X = data.drop('delay_minutes', axis=1)
        y = data['delay_minutes']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Saving processed data
        X_train.to_csv('X_train.csv', index=False)
        X_test.to_csv('X_test.csv', index=False)
        y_train.to_csv('y_train.csv', index=False)
        y_test.to_csv('y_test.csv', index=False)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
