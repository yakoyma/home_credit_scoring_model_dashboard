# Standard library
import platform
import json

# Other libraries
import flask
import pandas as pd
import joblib


from flask import Flask, request, jsonify
from joblib import load


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('JSON: {}'.format(json.__version__))
print('Flask: {}'.format(flask.__version__))
print('Pandas: {}'.format(pd.__version__))
print('Joblib: {}'.format(joblib.__version__))



app = Flask(__name__)
app.config['DEBUG'] = True


# Load the pre-trained ML model
MODEL_PATH = 'models/flaml/model.joblib'
model = load(MODEL_PATH)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """This function handles prediction requests using the Flask API.
    The endpoint accepts a JSON file containing a customer data, loads a
    pre-trained Machine Learning model, and returns the model prediction
    (predicted probability).

    Returns:
        result (float): app response containing the model prediction
                        (probability)
    """

    try:

        # Get the customer data from the dashboard request
        if not request.is_json:
            return jsonify({'error': 'Request must be in JSON format'}), 400

        customer_data = request.get_json()

        if not customer_data:
            return jsonify({'error': 'No data received'}), 400

        X_customer = pd.json_normalize(customer_data)

        # Make prediction
        y_proba = model.predict_proba(X_customer)[:, 1]
        probability = y_proba[0]

        # App response to the dashboard with the predicted probability
        return jsonify({'result': probability}), 200

    except Exception as error:
        app.logger.error(f'The following error occurred:\n{error}')
        return jsonify({'error': str(error)}), 500



if __name__ == '__main__':
    app.run(debug=True)
