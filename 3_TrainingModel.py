# 3_TrainingModel.py

import pandas as pd
import joblib
import logging
import numpy as np
import time
import requests  # If integrating with an external scheduling system
from sklearn.metrics import accuracy_score

# Configure logging
logging.basicConfig(filename='ai_scheduling.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def load_data_and_model(x_path, y_path, model_path):
    """
    Loads training data and a pre-tuned model from specified file paths.

    :param x_path: Path to the training features file.
    :param y_path: Path to the training labels file.
    :param model_path: Path to the saved pre-tuned model file.
    :return: Tuple of training features, labels, and the model, or (None, None, None) if an error occurs.
    """
    try:
        X_train = pd.read_csv(x_path)
        y_train = pd.read_csv(y_path)
        model = joblib.load(model_path)
        logging.info("Training data and model loaded successfully.")
        return X_train, y_train, model
    except Exception as e:
        logging.error(f"Error loading data or model: {e}")
        return None, None, None

def train_model(model, X_train, y_train):
    """
    Trains the model with the provided training data.

    :param model: The model to be trained.
    :param X_train: Training features.
    :param y_train: Training labels.
    :return: Trained model, or None if an error occurs during training.
    """
    try:
        model.fit(X_train, y_train.values.ravel())
        logging.info("Model training completed successfully.")
        return model
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        return None

def save_trained_model(model, filename):
    """
    Saves the trained model to a file.

    :param model: Trained model to save.
    :param filename: Filename to save the model.
    """
    try:
        joblib.dump(model, filename)
        logging.info(f"Trained model saved successfully as {filename}.")
    except Exception as e:
        logging.error(f"Error saving trained model: {e}")

def simulate_scenarios(X_train, y_train):
    """
    Augments the training data by simulating different scheduling scenarios.

    :param X_train: Training features.
    :param y_train: Training labels.
    :return: Augmented training features and labels.
    """
    # Define the number of scenarios to simulate
    num_scenarios = 100  # This can be adjusted based on your needs

    # Initialize arrays to hold simulated data
    X_train_simulated = np.copy(X_train)
    y_train_simulated = np.copy(y_train)

    # Loop to create simulated scenarios
    for _ in range(num_scenarios):
        # Randomly select data to simulate
        random_indices = np.random.choice(X_train.shape[0], size=int(X_train.shape[0] * 0.1), replace=False)
        simulated_data = X_train[random_indices, :]

        # Apply some modifications to simulate scenarios
        # Example: Randomly adjust 'departure_hour' and 'arrival_hour'
        for _ in range(100):  # Number of scenarios
            indices = np.random.choice(X_train.index, size=int(len(X_train) * 0.1), replace=False)
            X_scenario = X_train.loc[indices].copy()
            X_scenario['scheduled_departure_hour'] = np.random.choice(range(24), size=len(indices))
            X_scenario['scheduled_arrival_hour'] = (X_scenario['scheduled_departure_hour'] + np.random.choice(range(1, 5), size=len(indices))) % 24

        # Append simulated data to the training set
        X_train = pd.concat([X_train, X_scenario])
        y_train = pd.concat([y_train, y_train.loc[indices]])

    return X_train, y_train

def evaluate_model_during_training(model, X_val, y_val):
    # Evaluate model on a validation set
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    logging.info(f"Validation Accuracy: {accuracy}")
    return accuracy

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    # Training with periodic evaluation
    model.fit(X_train, y_train.values.ravel())
    evaluate_model_during_training(model, X_val, y_val)

def integrate_with_scheduling_system(model, X_data, scheduling_system_api):
    """
    Integrates the AI model with the existing flight scheduling system by sending predictions.

    :param model: The trained AI model.
    :param X_data: Data for which predictions are to be made.
    :param scheduling_system_api: API endpoint of the existing scheduling system.
    """
    try:
        # Generate predictions
        predictions = model.predict(X_data)

        # Send predictions to the scheduling system
        # This is a hypothetical POST request - adjust as needed for API
        response = requests.post(scheduling_system_api, json={'predictions': predictions.tolist()})
        
        if response.status_code == 200:
            logging.info("Predictions successfully integrated with the scheduling system.")
        else:
            logging.error(f"Failed to integrate with scheduling system. Status Code: {response.status_code}")
        
    except Exception as e:
        logging.error(f"Error during integration with scheduling system: {e}")


def continuous_learning(model, new_data_stream, retrain_interval):
    """
    Continuously updates the model with new data.

    :param model: The AI model.
    :param new_data_stream: Function or generator that yields new data.
    :param retrain_interval: Time interval (in seconds) between retraining sessions.
    """
    while True:
        try:
            # Wait for the specified interval
            time.sleep(retrain_interval)

            # Get new data
            new_data = next(new_data_stream)
            X_new, y_new = new_data['features'], new_data['labels']

            # Retrain the model with new data
            model.fit(X_new, y_new.values.ravel())
            logging.info("Model retrained with new data.")

            # Optionally, can save the retrained model
            # joblib.dump(model, 'retrained_model.pkl')

        except StopIteration:
            logging.info("No more new data available for retraining.")
            break
        except Exception as e:
            logging.error(f"Error during continuous learning: {e}")


# Main execution
if __name__ == "__main__":
    try:
        # Load data and split for validation
        X_train, y_train, model = load_data_and_model('X_train.csv', 'y_train.csv', 'decision_tree_model.pkl')
        # Existing code for loading model and data
        X_train, y_train = simulate_scenarios(X_train, y_train)
        if X_train is not None and y_train is not None:
            model = train_model(model, X_train, y_train)
            # Save the model based on improved performance criteria
            save_trained_model(model, 'trained_decision_tree_model.pkl')
    except Exception as e:
        logging.error(f"An error occurred in the training script: {e}")
