# 5_Server.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load the model. Ensure this is the correct model file.
model = joblib.load('optimized_random_forest_model.pkl')

def preprocess_input(data):
    """
    Preprocesses the input data to match the model's input format.
    """
    df = pd.DataFrame([data])

    # Handle datetime conversion and hour extraction
    try:
        df['scheduled_departure_hour'] = pd.to_datetime(df['scheduled_departure_time']).dt.hour
        df['scheduled_arrival_hour'] = pd.to_datetime(df['scheduled_arrival_time']).dt.hour
        df['departure_hour'] = pd.to_datetime(df['departureTime']).dt.hour
        df['arrival_hour'] = pd.to_datetime(df['arrivalTime']).dt.hour
    except KeyError as e:
        print(f"Missing key in data: {e}")
        return None

    # Encode and scale data
    label_encoder = LabelEncoder()
    scaler = MinMaxScaler()
    try:
        df['weather_condition_encoded'] = label_encoder.fit_transform(df['weather_condition'])
        numerical_cols = ['scheduled_departure_hour', 'scheduled_arrival_hour', 'departure_hour', 'arrival_hour']
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    except KeyError as e:
        print(f"Missing key for encoding/scaling: {e}")
        return None

    # Columns used by the model
    model_cols = ['weather_condition_encoded', 'scheduled_departure_hour', 'scheduled_arrival_hour']
    return df[model_cols]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    processed_data = preprocess_input(data)

    if processed_data is not None:
        # Make prediction
        try:
            prediction = model.predict(processed_data)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({'error': 'Error during prediction'}), 500

        # Format for Chart.js visualization
        chart_data = {
            'labels': ['Predicted Delay'],
            'datasets': [{
                'label': 'Delay in Minutes',
                'data': prediction.tolist(),
                'backgroundColor': ['rgba(255, 99, 132, 0.2)'],
                'borderColor': ['rgba(255, 99, 132, 1)'],
                'borderWidth': 1
            }]
        }
        
        return jsonify({'chartData': chart_data})
    else:
        return jsonify({'error': 'Invalid input data'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
