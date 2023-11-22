# 6_RealTimeDataProcessing.py

import requests
import time
import json
import logging
import pandas as pd
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
        data = response.json()
        logging.info("Real-time data fetched successfully.")
        return data
    except Exception as e:
        logging.error(f"Error fetching real-time data: {e}")
        return None

def process_data(data):
    """
    Processes the real-time data.

    :param data: The real-time data to be processed.
    :return: Processed data.
    """
    # Convert relevant columns to datetime
    data['scheduled_departure_time'] = pd.to_datetime(data['scheduled_departure_time'])
    data['scheduled_arrival_time'] = pd.to_datetime(data['scheduled_arrival_time'])

    # Extract features used by the model
    data['scheduled_departure_hour'] = data['scheduled_departure_time'].dt.hour
    data['scheduled_arrival_hour'] = data['scheduled_arrival_time'].dt.hour

    # Encode categorical variables and scale numerical columns as per model requirements
    label_encoder = LabelEncoder()
    scaler = MinMaxScaler()
    data['weather_condition_encoded'] = label_encoder.fit_transform(data['weather_condition'])
    data[['scheduled_departure_hour', 'scheduled_arrival_hour']] = scaler.fit_transform(data[['scheduled_departure_hour', 'scheduled_arrival_hour']])
    
    return data

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
            processed_data = preprocess_realtime_data(data)
            send_to_scheduling_system(processed_data, scheduling_system_endpoint, model)

        # Fetch new data at regular intervals (e.g., every 10 seconds)
        time.sleep(10)

def main():
    source_url = 'http://realtime_data_source_url'
    scheduling_system_endpoint = 'http://scheduling_system_endpoint'

    while True:
        # Fetch real-time data
        data = fetch_realtime_data(source_url)
        if data:
            # Process the data
            processed_data = process_data(data)
            # Send processed data to scheduling system
            send_to_scheduling_system(processed_data, scheduling_system_endpoint)

        # Example: Fetch new data every 10 seconds
        time.sleep(10)

if __name__ == '__main__':
    model_path = 'path_to_trained_model.pkl'
    source_url = 'http://realtime_data_source_url'
    scheduling_system_endpoint = 'http://scheduling_system_endpoint'
    main(model_path, source_url, scheduling_system_endpoint)
