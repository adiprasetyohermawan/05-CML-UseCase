import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import io
import base64
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

# Load the files (model, scaler, df_result)
scaler = joblib.load('model/scaler.save')
model = joblib.load('model/pii-customer_kmeans_model.joblib')
df_visualize = pd.read_csv("../df_visualize.csv")
df_analysis = df_visualize[['bank_id', 'dob', 'city', 'total_transaction', 'income', 'cluster']]

# df_analysis = pd.DataFrame({
#     "bank_id": [7, 8, 9, 7, 8],
#     "dob": [1, 2, 3, 2, 1],
#     "city": [1, 4, 3, 2, 5],
#     "total_transaction": [731064773.0, 787979424.0, 207590810.0, 870578029.0, 686373810.0],
#     "income": [7652685.0, 9305611.0, 3984638.0, 2603126.0, 8529827.0],
#     "cluster": [3, 2, 0, 3, 0]
# })

# Calculate cluster summaries and store them for the visualization page
summary = {}
for i in range(4):  # Assuming 4 clusters
    summary[i] = df_analysis[df_analysis['cluster'] == i].describe().T.to_dict()

# Encoding functions
def dob_encode(dob):
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

    # Find the correct age range
    for age_range, label in age_range_mapping.items():
        if age_range[0] <= dob <= age_range[1]:
            # Return the encoded integer for the age range
            return age_mapping.get(label, 0)
    
    # Return 0 if dob doesn't fall within any defined range
    return 0

def bank_encode(bank_id):
    bank_mapping = {
        '002': 1, '008': 2, '009': 3, '011': 4,
        '013': 5, '014': 6, '022': 7
    }
    return bank_mapping.get(bank_id, 0)

def city_encode(city):
    city_mapping = {
        'Bandung': 1, 'Jakarta': 2, 'Medan': 3,
        'Semarang': 4, 'Surabaya': 5
    }
    return city_mapping.get(city, 0)

# Interpretations for each cluster
cluster_interpretations = {
    0: "If assigned to Cluster 0, the new customer likely has characteristics of a mature, financially active individual with high-value transactions.",
    1: "If assigned to Cluster 1, the new customer is likely an older, financially active individual, mostly based in Jakarta, with established banking relationships and higher incomes.",
    2: "If assigned to Cluster 2, the new customer is likely a younger, active customer with moderate to high incomes, residing in areas like Semarang or Jakarta, and engaging in significant transaction volumes.",
    3: "If assigned to Cluster 3, the new customer is likely a young professional in their late 20s, primarily in Jakarta, with diverse transaction levels and stable to high income."
}

# Predict customer cluster
def pred_cust_cluster(data):
    dob = int(data['dob'])
    dob = dob_encode(dob)
    bank_id = bank_encode(data['bank_id'])
    city = city_encode(data['city'])
    total_transaction = data['total_transaction']
    income = data['income']
    
    print({dob, bank_id, city, total_transaction, income})

    features = [dob, bank_id, city, total_transaction, income]
    scaled_features = scaler.transform(np.array(features).reshape(1, -1))
    result = model.predict(scaled_features)

    # Get the interpretation for the predicted cluster
    interpretation = cluster_interpretations.get(result.item(), "No interpretation available.")
    return {"customer_cluster": result.item(), "interpretation": interpretation}

# Route for the main page
@app.route('/')
def index():
    return render_template('predict.html')

# Route to get summary stats for each cluster
@app.route('/get_cluster_summary')
def get_cluster_summary():
    summary_stats = {
        int(cluster): df_analysis[df_analysis['cluster'] == cluster].describe().to_dict()
        for cluster in df_analysis['cluster'].unique()
    }
    # debug summary_stats
    # print(summary_stats)
    return jsonify(summary_stats)

# Route to generate histogram for a specific cluster
@app.route('/get_histogram/<int:cluster_id>')
def get_histogram(cluster_id):
    # Filter data for the specified cluster
    cluster_data = df_analysis[df_analysis['cluster'] == cluster_id]

    # Generate histogram
    fig, axs = plt.subplots(1, len(cluster_data.columns) - 1, figsize=(18, 5))
    # fig.suptitle(f'Cluster {cluster_id} - Histogram')

    # Plot histograms for each feature in the cluster
    for i, column in enumerate(cluster_data.columns[:-1]):  # Exclude 'cluster' column
        cluster_data[column].plot(kind='hist', ax=axs[i], title=column, bins=20, color='#4c72b0')

    # Convert plot to PNG image and encode it in base64
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)

    return jsonify({'img_data': img})

# Prediction API route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = pred_cust_cluster(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5002, debug=True)