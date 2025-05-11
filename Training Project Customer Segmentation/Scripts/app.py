import os
import requests
import joblib
import numpy as np
import flask

from flask import Flask, request, render_template
from datetime import datetime

app = Flask(__name__)

# Load the pre-trained KMeans model
model = joblib.load('model/pii-customer_kmeans_model.joblib')
scaler = joblib.load('model/scaler.save')

# Interpretations for each cluster
cluster_interpretations = {
    0: "If assigned to Cluster 0, the new customer likely has characteristics of a mature, financially active individual with high-value transactions.",
    1: "If assigned to Cluster 1, the new customer is likely an older, financially active individual, mostly based in Jakarta, with established banking relationships and higher incomes.",
    2: "If assigned to Cluster 2, the new customer is likely a younger, active customer with moderate to high incomes, residing in areas like Semarang or Jakarta, and engaging in significant transaction volumes.",
    3: "If assigned to Cluster 3, the new customer is likely a young professional in their late 20s, primarily in Jakarta, with diverse transaction levels and stable to high income."
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from the form
    age = request.form["age"]
    bank_id = request.form["bank_id"]
    city = request.form["city"]
    total_transaction = request.form["total_transaction"]
    income = request.form["income"]
    
    # Encoding rules
    age_range_mapping = {
        (19, 22): '19 - 22', (23, 26): '23 - 26', (27, 30): '27 - 30',
        (31, 34): '31 - 34', (35, 38): '35 - 38', (39, 42): '39 - 42', 
        (43, 46): '43 - 46', (47, 50): '47 - 50', (51, 54): '51 - 54'
    }
    age_mapping = {
        '19 - 22': 1, '23 - 26': 2, '27 - 30': 3,
        '31 - 34': 4, '35 - 38': 5, '39 - 42': 6,
        '43 - 46': 7, '47 - 50': 8, '51 - 54': 9
    }
    bank_mapping = {
        '002': 1, '008': 2, '009': 3, '011': 4, 
        '013': 5, '014': 6, '022': 7
    }
    city_mapping = {
        'Bandung': 1, 'Jakarta': 2, 'Medan': 3, 
        'Semarang': 4, 'Surabaya': 5
    }
    
    age_range_encoded = age_range_mapping.get(age, 0)
    bank_encoded = bank_mapping.get(bank_id, 0)
    city_encoded = city_mapping.get(city, 0)
    total_transaction = float(total_transaction)
    income = float(income)

    # Feature set up in Flask Local
    age_encoded = age_mapping.get(age_range_encoded, 0)
    features = [age_encoded, bank_encoded, city_encoded, total_transaction, income]
    scaled_features = scaler \
        .transform(np.array(features) \
        .reshape(1, -1))
    result = model.predict(scaled_features)
    return render_template('result.html', predicted_cluster=result.item())

    # Feature set up in Flask CML
    # Prepare the data payload as a JSON string
    # data_payload = '{"accessKey":"mp0gnwj6mlecptq77wvrtx3k5yw9rd4w","request":{"total_transaction":' + str(total_transaction) + ',"income":' + str(income) + ',"dob":"' + dob + '"}}'

    # Send POST request to the model endpoint
    # response = requests.post(
    #     'https://modelservice.cml.apps.lintas.cloudeka.ai/model',
    #     data=data_payload,
    #     headers={
    #         'Content-Type': 'application/json',
    #         'Authorization': 'Bearer 736074ca2f6a91768372d7002165ff40dc9d594c832d14e869386ac9be725995.0a8592b8711622330b01d9e38c8b41e89b58c594ff0798cd9641d25bae9a8044'
    #     }
    # )

    # Check the response status
    # if response.status_code == 200:
    #     prediction_data = response.json()

    #     # Check if the response indicates success
    #     if prediction_data.get('success'):
    #         response_data = prediction_data['response']
    #         predicted_cluster = response_data.get('predicted_cluster', 'N/A')
    #         interpretation = response_data.get('interpretation', 'No interpretation available.')

    #         return render_template("result.html", predicted_cluster=predicted_cluster, interpretation=interpretation)
    #     else:
    #         return render_template("error.html", error="Prediction was not successful.")
    # else:
    #     return render_template("error.html", error=response.text)


if __name__ == "__main__":
    # port = int(os.environ.get("PORT",8080))
    # app.run(host='127.0.0.1', port=int(os.environ['CDSW_APP_PORT']))
    app.run(debug=True, port=5001)