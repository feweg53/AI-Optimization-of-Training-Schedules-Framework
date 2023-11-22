# 6_RealTimeDataProcessing.py

import requests
import time
import json
import logging
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Configure logging
logging.basicConfig(filename='realtime_data_processing.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def fetch_realtime_data(source_url):
    """
    Fetches data from the real-time data source.

    :param source_url: URL or endpoint of the real-time data source.
    :return: Data fetched from the source.
    """
    try:
        response = requests.get(source_url)
        if response.status_code == 200:
            data = response.json()
            logging.info("Real-time data fetched successfully.")
            return data
        else:
            logging.error(f"Failed to fetch data. Status Code: {response.status_code}")
            return None
    except Exception as e:
            logging.error(f"Error fetching real-time data: {e}")
            return None

def process_data(data):
    """
    Processes the real-time data.

    :param data: The real-time data to be processed.
    :return: Processed data.
    """
    df = pd.DataFrame([data])

    # Convert relevant columns to datetime
    df['scheduled_departure_time'] = pd.to_datetime(df['scheduled_departure_time'])
    df['scheduled_arrival_time'] = pd.to_datetime(df['scheduled_arrival_time'])

    # Extract features used by the model
    df['scheduled_departure_hour'] = df['scheduled_departure_time'].dt.hour
    df['scheduled_arrival_hour'] = df['scheduled_arrival_time'].dt.hour

    # Encode and scale as per model's preprocessing requirements
    # Note: Use saved LabelEncoder and Scaler objects from the training phase
    # For simplicity, using fit_transform here
    label_encoder = LabelEncoder()
    scaler = MinMaxScaler()
    df['weather_condition_encoded'] = label_encoder.fit_transform(df['weather_condition'])
    df[['scheduled_departure_hour', 'scheduled_arrival_hour']] = scaler.fit_transform(df[['scheduled_departure_hour', 'scheduled_arrival_hour']])

    # Selecting the columns used by the model
    model_cols = ['weather_condition_encoded', 'scheduled_departure_hour', 'scheduled_arrival_hour']
    return df[model_cols]

def send_to_scheduling_system(processed_data, scheduling_system_endpoint):
    """
    Sends the processed data to the scheduling system.

    :param processed_data: Data processed for scheduling.
    :param scheduling_system_endpoint: Endpoint of the scheduling system.
    """
    try:
        # Generate predictions
        predictions = model.predict(processed_data)
        response = requests.post(scheduling_system_endpoint, json={'predictions': predictions.tolist()})

        if response.status_code == 200:
            logging.info("Predictions successfully sent to the scheduling system.")
        else:
            logging.error(f"Failed to send predictions. Status Code: {response.status_code}")
    except Exception as e:
        logging.error(f"Error in sending predictions: {e}")

def main(model_path, source_url, scheduling_system_endpoint):
    # Load the trained model
    model = joblib.load(model_path)

    while True:
        # Fetch and process real-time data
        data = fetch_realtime_data(source_url)
        if data:
            processed_data = preprocess_realtime_data(data, model)
            send_to_scheduling_system(processed_data, scheduling_system_endpoint, model)

        # Fetch new data at regular intervals (e.g., every 10 seconds)
        time.sleep(10)

if __name__ == '__main__':
    model_path = 'path_to_trained_model.pkl'
    source_url = 'http://realtime_data_source_url'
    scheduling_system_endpoint = 'http://scheduling_system_endpoint'
    main(model_path, source_url, scheduling_system_endpoint)
