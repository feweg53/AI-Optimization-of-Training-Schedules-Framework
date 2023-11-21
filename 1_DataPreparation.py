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
    data.fillna(method='ffill', inplace=True)
    data['scheduled_departure_time'] = pd.to_datetime(data['scheduled_departure_time'])
    data['scheduled_arrival_time'] = pd.to_datetime(data['scheduled_arrival_time'])
    data['departure_hour'] = data['scheduled_departure_time'].dt.hour
    data['arrival_hour'] = data['scheduled_arrival_time'].dt.hour
    return data

def handle_uncertainties(data):
    """
    Adjusts the dataset to handle uncertainties such as variable flight times 
    and rapidly changing conditions.

    :param data: DataFrame containing the schedule data.
    :return: Adjusted DataFrame.
    """
    # Example: Adjust flight times based on weather conditions
    # Assuming there's a 'weather_condition' column and it affects flight duration
    # You can adjust this logic based on your specific data and requirements

    # Define a simple mapping of weather conditions to delay factor
    weather_delay_factor = {
        'clear': 1.0,   # no delay
        'rainy': 1.1,   # 10% more time
        'stormy': 1.2,  # 20% more time
        'foggy': 1.15,  # 15% more time
    }

    # Apply the delay factor to the flight duration
    data['adjusted_flight_duration'] = data.apply(
        lambda row: row['flight_duration'] * weather_delay_factor.get(row['weather_condition'], 1),
        axis=1
    )

    # Example: Handling crew availability
    # Assuming a column 'crew_on_standby' indicating if extra crew is available
    # Adjusting flight status based on crew availability
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
    label_encoder = LabelEncoder()
    for col in categorical_columns:
        data[col] = label_encoder.fit_transform(data[col])
    scaler = MinMaxScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    return data

# Main execution
if __name__ == "__main__":
    try:
        # Load the data
        data = load_data('your_dataset.csv')
        if data is None:
            raise ValueError("No data loaded. Check if the file exists or is valid.")

        # Data preprocessing
        data = preprocess_data(data)
        logging.info("Data preprocessing completed.")
        data = handle_uncertainties(data)

        # Encoding and scaling
        categorical_columns = ['weather_conditions']
        numerical_columns = ['departure_hour', 'arrival_hour', 'other_numerical_column']
        data = encode_and_scale(data, categorical_columns, numerical_columns)
        logging.info("Encoding and scaling of data completed.")

        # Splitting the data
        X = data.drop('delay_minutes', axis=1)
        y = data['delay_minutes']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Data split into training and testing sets.")

        # Saving the processed data
        X_train.to_csv('X_train.csv', index=False)
        X_test.to_csv('X_test.csv', index=False)
        y_train.to_csv('y_train.csv', index=False)
        y_test.to_csv('y_test.csv', index=False)
        logging.info("Processed data saved successfully.")

    except Exception as e:
        logging.error(f"An error occurred during the main execution: {e}")
