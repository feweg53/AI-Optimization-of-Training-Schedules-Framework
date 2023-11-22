# 4_Testing.py

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
import logging
import shap  # Assuming SHAP is installed
# import LIME if want to use LIME
import matplotlib.pyplot as plt # used to create a heatmap for the confusion matrix.
import seaborn as sns
from sklearn.metrics import confusion_matrix # A confusion matrix is generated

# Configure logging
logging.basicConfig(filename='ai_scheduling.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def dynamic_scenario_generation(X_test):
    """
    Generates dynamic test scenarios based on different flight conditions.

    :param X_test: The test feature dataframe.
    :return: A dictionary of test scenarios.
    """
    scenarios = {}

    # Scenario based on different weather conditions
    for condition in X_test['weather_condition'].unique():
        scenarios[f'weather_{condition}'] = X_test[X_test['weather_condition'] == condition]

    # Scenario based on different times of day
    hours = range(0, 24, 6)  # Dividing the day into 4 parts (00-06, 06-12, etc.)
    for start_hour in hours:
        end_hour = (start_hour + 6) % 24
        time_frame = X_test['departure_hour'].between(start_hour, end_hour)
        scenarios[f'time_{start_hour}-{end_hour}'] = X_test[time_frame]

    # Additional scenarios can be added here based on other conditions

    return scenarios...

def perform_error_analysis(model, X_test, y_test, y_pred):
    """
    Performs error analysis on the model's predictions.

    :param model: Trained model.
    :param X_test: Test features.
    :param y_test: Actual labels of the test set.
    :param y_pred: Predicted labels of the test set.
    :return: Analysis results.
    """
    # Initialize a DataFrame to store error details
    error_df = X_test.copy()
    error_df['actual'] = y_test
    error_df['predicted'] = y_pred
    error_df['error'] = error_df['actual'] != error_df['predicted']

    # Analysis 1: Overall Error Rate
    overall_error_rate = error_df['error'].mean()
    print(f"Overall Error Rate: {overall_error_rate:.2%}")

    # Analysis 2: Error Distribution by Weather Condition
    error_by_weather = error_df.groupby('weather_condition')['error'].mean()
    print("\nError Distribution by Weather Condition:")
    print(error_by_weather)

    # Analysis 3: Error Distribution by Time of Day
    error_by_time = error_df.groupby('departure_hour')['error'].mean()
    print("\nError Distribution by Time of Day:")
    print(error_by_time)

    # Additional analyses can be performed based on other features

    # Example: Top 5 instances with highest error
    if 'probability' in error_df.columns:
        top_errors = error_df[error_df['error']].sort_values(by='probability', ascending=False).head(5)
        print("\nTop 5 instances with highest prediction error:")
        print(top_errors)

    return error_df

def visualize_model_performance(y_test, y_pred):
    """
    Visualizes the model's performance.

    :param y_test: Actual labels of the test set.
    :param y_pred: Predicted labels of the test set.
    """
    # Generate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plotting using seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def automated_model_diagnostics(model, X_test, y_test):
    """
    Performs automated diagnostics of the model.

    :param model: The trained machine learning model.
    :param X_test: Test features for the model.
    :param y_test: True labels for the test data.
    :return: A dictionary with diagnostic results.
    """
    diagnostics = {}

    # Model accuracy
    y_pred = model.predict(X_test)
    diagnostics['accuracy'] = accuracy_score(y_test, y_pred)
    diagnostics['classification_report'] = classification_report(y_test, y_pred)

    # Feature importance (specific to tree-based models like RandomForest)
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.Series(model.feature_importances_, index=X_test.columns)
        diagnostics['feature_importance'] = feature_importance.sort_values(ascending=False)

    return diagnostics

def evaluate_under_scenarios(model, X_test, y_test):
    """
    Evaluates the model under different test scenarios to check robustness.

    :param model: Trained model.
    :param X_test: Test features.
    :param y_test: Test labels.
    :return: Evaluation results under different scenarios.
    """
    # Define specific scenarios to test the model's performance
    # For example, testing the model's performance during different times of day
    # or under varying weather conditions
    scenarios = {
        'clear_weather': X_test[X_test['weather_condition'] == 'clear'],
        'stormy_weather': X_test[X_test['weather_condition'] == 'stormy']
    }

    # This is a placeholder implementation, should be adapted to use case
    results = {}
    for scenario_name, X_scenario in scenarios.items():
        y_pred = model.predict(X_scenario)
        results[scenario_name] = {
            "accuracy": accuracy_score(y_test.loc[X_scenario.index], y_pred),
            "report": classification_report(y_test.loc[X_scenario.index], y_pred)
        }

    return results

def evaluate_model(y_test, y_pred):
    """
    Evaluates the model's performance.

    :param y_test: Actual labels of the test set.
    :param y_pred: Predicted labels of the test set.
    :return: Dictionary containing accuracy and classification report.
    """
    try:
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logging.info("Model evaluation completed successfully.")
        return {"accuracy": accuracy, "report": report}
    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        return None

def perform_predictions(model, X_test):
    """
    Performs predictions using the provided model on the test set.

    :param model: Trained machine learning model.
    :param X_test: Test features.
    :return: Predictions array.
    """
    try:
        predictions = model.predict(X_test)
        logging.info("Predictions performed successfully.")
        return predictions
    except Exception as e:
        logging.error(f"Error during predictions: {e}")
        return None

def generate_explanation(model, data_point):
    """
    Generates an explanation for a specific prediction.

    :param model: The AI model.
    :param data_point: The data point for which an explanation is needed.
    :return: Explanation of the model's decision.
    """
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_point)
    return shap.force_plot(explainer.expected_value, shap_values, data_point)

# Main execution
if __name__ == "__main__":
    try:
        # Load test data and model
        X_test, y_test, model = load_test_data_and_model('X_test.csv', 'y_test.csv', 'optimized_random_forest_model.pkl')

        # Perform predictions - called once
        y_pred = perform_predictions(model, X_test)

        if y_pred is not None:
            # Evaluate model under different scenarios
            scenario_results = evaluate_under_scenarios(model, X_test, y_test)
            print(f"Scenario Evaluation Results: {scenario_results}")

            # Evaluate model accuracy and classification report
            evaluation_results = evaluate_model(y_test, y_pred)
            print(f"Model Accuracy: {evaluation_results['accuracy']}")
            print(f"Classification Report:\n{evaluation_results['report']}")

            # Error Analysis
            perform_error_analysis(model, X_test, y_test, y_pred)

            # Visualization of model performance
            visualize_model_performance(y_test, y_pred)

            # Model Diagnostics
            diagnostics = automated_model_diagnostics(model, X_test, y_test)
            print(f"Model Diagnostics: {diagnostics}")

            # Model explanation for a sample
            sample_data_point = X_test.iloc[0:5]  # Sample data points for explanation
            explanation = generate_explanation(model, sample_data_point)
            print(f"Model Explanation: {explanation}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
