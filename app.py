import flask
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

import joblib
from google.colab import files

# Save the model to a file
model_filename = 'balanced_logistic_regression_model.joblib'
joblib.dump(balanced_lr, model_filename)
print(f"Model successfully saved as {model_filename}")

# Trigger the download to your local machine
files.download(model_filename)
# Load the saved model and scaler
model = joblib.load('balanced_logistic_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)
        
        # Convert data into a pandas DataFrame (expects the same columns as X_train)
        # Expected keys: 'Type', 'Air temperature [K]', 'Process temperature [K]', 
        # 'Rotational speed [rpm]', 'Torque [Nm]', 'Vibration Levels', 'Operational Hours'
        input_df = pd.DataFrame([data])
        
        # Scale the 'Rotational speed [rpm]' feature using our loaded scaler
        list1 = ['Rotational speed [rpm]']
        input_df[list1] = scaler.transform(input_df[list1])
        
        # Ensure columns are ordered correctly before calling predict
        feature_order = [
            'Type', 
            'Air temperature [K]', 
            'Process temperature [K]', 
            'Rotational speed [rpm]', 
            'Torque [Nm]', 
            'Vibration Levels', 
            'Operational Hours'
        ]
        input_df = input_df[feature_order]
        
        # Perform prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        # Get confidence mapping for classes
        classes = model.classes_
        probabilities_dict = {classes[i]: float(prediction_proba[i]) for i in range(len(classes))}
        
        return jsonify({
            'status': 'success',
            'prediction': str(prediction),
            'probabilities': probabilities_dict
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    print("Flask API Server loaded. Ready for predictions!")
