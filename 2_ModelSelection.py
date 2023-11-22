# 2_ModelSelection.py

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  # Additional model type
from sklearn.model_selection import GridSearchCV
import joblib
import logging

# Configure logging
logging.basicConfig(filename='ai_scheduling.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def load_training_data(x_path, y_path):
    """
    Loads training data from specified file paths.

    :param x_path: Path to the training features file.
    :param y_path: Path to the training labels file.
    :return: Tuple of training features and labels, or (None, None) if an error occurs.
    """
    try:
        X_train = pd.read_csv(x_path)
        y_train = pd.read_csv(y_path)
        logging.info("Training data loaded successfully.")
        return X_train, y_train
    except Exception as e:
        logging.error(f"Error loading training data: {e}")
        return None, None

def initialize_model(model_type='decision_tree', hyperparameters=None):
    """
    Initializes a machine learning model based on the specified type and hyperparameters.

    :param model_type: Type of model to initialize.
    :param hyperparameters: Hyperparameters for the model.
    :return: Initialized model or None if the model type is not supported.
    """
    if model_type == 'decision_tree':
        model = DecisionTreeClassifier(**hyperparameters) if hyperparameters else DecisionTreeClassifier()
    elif model_type == 'random_forest':
        model = RandomForestClassifier(**hyperparameters) if hyperparameters else RandomForestClassifier()
    else:
        logging.error(f"Model type '{model_type}' is not supported.")
        return None
    return model

def perform_hyperparameter_tuning(model, param_grid, X_train, y_train):
    """
    Performs hyperparameter tuning using GridSearchCV.

    :param model: Model to tune.
    :param param_grid: Grid of parameters to search.
    :param X_train: Training features.
    :param y_train: Training labels.
    :return: Model with the best parameters from GridSearchCV.
    """
    model = RandomForestClassifier()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    # The GridSearchCV is set to use n_jobs=-1 to utilize all available CPU cores for faster execution.
    # Added verbose=1 to GridSearchCV for more detailed output during the model training process.
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train.values.ravel())
    logging.info(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

def save_model(model, filename):
    """
    Saves the model to a file.

    :param model: Model to save.
    :param filename: Filename to save the model.
    """
    try:
        joblib.dump(model, filename)
        logging.info(f"Model saved successfully as {filename}.")
    except Exception as e:
        logging.error(f"Error saving model: {e}")

# Main execution
if __name__ == "__main__":
    try:
        X_train, y_train = load_training_data('X_train.csv', 'y_train.csv')
        if X_train is not None and y_train is not None:
            # Perform hyperparameter tuning
            optimized_model = perform_hyperparameter_tuning(X_train, y_train)
            # Save the best model
            save_model(optimized_model, 'optimized_random_forest_model.pkl')
    except Exception as e:
        logging.error(f"An error occurred in model selection: {e}")
