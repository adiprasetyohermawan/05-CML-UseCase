import os
import requests
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from the form
    total_transaction = request.form["total_transaction"]
    income = request.form["income"]
    dob = request.form["dob"]

    # Prepare the data payload as a JSON string
    data_payload = '{"accessKey":"mp0gnwj6mlecptq77wvrtx3k5yw9rd4w","request":{"total_transaction":' + str(total_transaction) + ',"income":' + str(income) + ',"dob":"' + dob + '"}}'

    # Send POST request to the model endpoint
    response = requests.post(
        'https://modelservice.cml.apps.lintas.cloudeka.ai/model',
        data=data_payload,
        headers={
            'Content-Type': 'application/json',
            'Authorization': 'Bearer 736074ca2f6a91768372d7002165ff40dc9d594c832d14e869386ac9be725995.0a8592b8711622330b01d9e38c8b41e89b58c594ff0798cd9641d25bae9a8044'
        }
    )

    # Check the response status
    if response.status_code == 200:
        prediction_data = response.json()

        # Check if the response indicates success
        if prediction_data.get('success'):
            response_data = prediction_data['response']
            predicted_cluster = response_data.get('predicted_cluster', 'N/A')
            interpretation = response_data.get('interpretation', 'No interpretation available.')

            return render_template("result.html", predicted_cluster=predicted_cluster, interpretation=interpretation)
        else:
            return render_template("error.html", error="Prediction was not successful.")
    else:
        return render_template("error.html", error=response.text)


if __name__ == "__main__":
    port = int(os.environ.get("PORT",8080))
    app.run(host='127.0.0.1', port=int(os.environ['CDSW_APP_PORT']))
