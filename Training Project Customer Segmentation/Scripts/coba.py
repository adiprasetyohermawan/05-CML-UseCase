from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load pre-trained model (e.g., KMeans) and scaler if needed
model = joblib.load("model/pii-customer_kmeans_model.joblib")  # Replace with actual model path
scaler = joblib.load("model/scaler.save")  # Assuming features were scaled before clustering

# Dummy Data for Cluster Summary (Replace with actual data loading)
cluster_summary = {
    "Cluster 1": {"Count": 100, "Avg Income": 50000, "Avg Transaction": 200},
    "Cluster 2": {"Count": 80, "Avg Income": 70000, "Avg Transaction": 150},
    # Add more clusters as needed
}

@app.route('/')
def index():
    return render_template('coba.html', cluster_summary=cluster_summary)

# Visualization Function
@app.route('/visualize_clusters')
def visualize_clusters():
    # Assume data is loaded and processed to show histogram and summary
    # For example, creating a histogram of income by cluster
    # Replace with real data and plot generation logic
    fig, ax = plt.subplots()
    data = pd.DataFrame({"Cluster": ["Cluster 1", "Cluster 2"], "Income": [50000, 70000]})
    data.plot(kind="bar", x="Cluster", y="Income", ax=ax)
    
    # Save to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    
    return jsonify({'img_data': img})

# Prediction Function
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    age = data['age']
    bank_id = data['bank_id']
    city = data['city']
    total_transaction = data['total_transaction']
    income = data['income']
    
    # Convert input to model's expected format
    features = np.array([[age, bank_id, city, total_transaction, income]])
    features_scaled = scaler.transform(features)  # Scale features if needed
    
    # Predict cluster
    cluster = model.predict(features_scaled)[0]
    interpretation = f"The customer likely belongs to Cluster {cluster + 1}, characterized by ... (Add interpretation here)"
    
    return jsonify({'cluster': int(cluster), 'interpretation': interpretation})

if __name__ == '__main__':
    app.run(debug=True)