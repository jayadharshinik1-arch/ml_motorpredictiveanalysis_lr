import os

import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the saved model and scaler once at startup
model = joblib.load(os.path.join(BASE_DIR, 'balanced_logistic_regression_model.joblib'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.joblib'))

# Use the exact column names/order the model was trained with
# (note: the trained name is 'Vibration Levels ' with a trailing space)
FEATURE_ORDER = list(model.feature_names_in_)


@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'expected_features': [f.strip() for f in FEATURE_ORDER]})


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Tolerate keys with/without trailing spaces
        data = {str(k).strip(): v for k, v in data.items()}
        input_df = pd.DataFrame([data])
        input_df.columns = [c.strip() for c in input_df.columns]

        # Scale the 'Rotational speed [rpm]' feature using the loaded scaler
        scale_cols = list(scaler.feature_names_in_)
        input_df[scale_cols] = scaler.transform(input_df[scale_cols])

        # Reorder to the trained column names
        stripped_to_trained = {f.strip(): f for f in FEATURE_ORDER}
        missing = [k for k in stripped_to_trained if k not in input_df.columns]
        if missing:
            return jsonify({'status': 'error', 'message': f'Missing fields: {missing}'}), 400
        input_df = input_df[list(stripped_to_trained.keys())]
        input_df.columns = [stripped_to_trained[c] for c in input_df.columns]

        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        probabilities = {str(c): float(p) for c, p in zip(model.classes_, proba)}

        return jsonify({
            'status': 'success',
            'prediction': str(prediction),
            'probabilities': probabilities,
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
