import pandas as pd
import numpy as np
import json
import joblib
import argparse
from sklearn.preprocessing import MinMaxScaler

# Load the model and scaler
scaler_filename = 'scaler.save'
scaler = joblib.load('model/' + scaler_filename)
model_filename = 'pii-customer_kmeans_model.joblib'
model = joblib.load('model/' + model_filename)

# Encoding functions
def dob_encode(dob):
    age_mapping = {
        '19 - 22': 1, '23 - 26': 2, '27 - 30': 3,
        '31 - 34': 4, '35 - 38': 5, '39 - 42': 6,
        '43 - 46': 7, '47 - 50': 8, '51 - 54': 9
    }
    return age_mapping.get(dob, 0)

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
def pred_cust_cluster(args):
    dob = dob_encode(args['dob'])
    bank_id = bank_encode(args['bank_id'])
    city = city_encode(args['city'])
    total_transaction = args['total_transaction']
    income = args['income']
    
    features = [dob, bank_id, city, total_transaction, income]
    scaled_features = scaler.transform(np.array(features).reshape(1, -1))
    result = model.predict(scaled_features)

    # Get the interpretation for the predicted cluster
    interpretation = cluster_interpretations.get(result.item(), "No interpretation available.")

    return {
        "customer_cluster": result.item(),
        "interpretation": interpretation
    }

# Main function to get input from command line
def main():
    parser = argparse.ArgumentParser(description="Predict customer cluster based on input data.")
    parser.add_argument('--dob', type=str, required=True, help="Age group of the customer, e.g., '19 - 22'")
    parser.add_argument('--bank_id', type=str, required=True, help="Bank ID of the customer, e.g., '022'")
    parser.add_argument('--city', type=str, required=True, help="City of the customer, e.g., 'Jakarta'")
    parser.add_argument('--total_transaction', type=int, required=True, help="Total transaction amount")
    parser.add_argument('--income', type=int, required=True, help="Total income of the customer")
    
    args = parser.parse_args()
    input_data = {
        "dob": args.dob,
        "bank_id": args.bank_id,
        "city": args.city,
        "total_transaction": args.total_transaction,
        "income": args.income
    }
    
    # Predict cluster
    output = pred_cust_cluster(input_data)
    print("Prediction result:", output)

if __name__ == "__main__":
    main()

# How to run predict.py
# python predict.py --dob '19 - 22' --bank_id '022' --city 'Jakarta' --total_transaction 731064773 --income 7652685