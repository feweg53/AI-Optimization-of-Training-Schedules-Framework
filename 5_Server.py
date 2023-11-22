# 5_Server.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

app = Flask(__name__)

# Load the model. Ensure this is the correct model file.
model = joblib.load('optimized_random_forest_model.pkl')  

def preprocess_input(data):
    """
    Preprocesses the input data to match the model's input format.
    Convert the incoming JSON data into the format expected by the model.
    """
    df = pd.DataFrame([data])

    # Convert time columns to datetime and extract hours
    df['scheduled_departure_time'] = pd.to_datetime(df['scheduled_departure_time'])
    df['scheduled_arrival_time'] = pd.to_datetime(df['scheduled_arrival_time'])
    df['scheduled_departure_hour'] = df['scheduled_departure_time'].dt.hour
    df['scheduled_arrival_hour'] = df['scheduled_arrival_time'].dt.hour

    # Encode categorical variables and scale numerical columns
    label_encoder = LabelEncoder()
    scaler = MinMaxScaler()
    df['weather_condition_encoded'] = label_encoder.fit_transform(df['weather_condition'])
    numerical_cols = ['scheduled_departure_hour', 'scheduled_arrival_hour']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Select the columns used by the model
    model_cols = ['weather_condition_encoded', 'scheduled_departure_hour', 'scheduled_arrival_hour']
    return df[model_cols]

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    processed_data = preprocess_input(data)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, port=5000) #set debug=False for production deployment
