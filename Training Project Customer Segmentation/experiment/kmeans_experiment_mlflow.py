import sys
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.sklearn
import os
import json

def update_colnames(df):
    updated_columns = [col.replace('customer.', '') for col in df.columns]
    df.columns = updated_columns
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['email', 'phone_numb', 'account_number', 'ktp', 'address', 'postal_code'])
    string_columns = ['cust_id', 'name', 'bank_id', 'dob', 'city']
    df[string_columns] = df[string_columns].astype('string')
    decimal_columns = ['total_transaction', 'income']
    df[decimal_columns] = df[decimal_columns].astype('float64')
    return df

def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    age_mapping = {'19 - 22': 1, '23 - 26': 2, '27 - 30': 3, '31 - 34': 4, '35 - 38': 5,
                   '39 - 42': 6, '43 - 46': 7, '47 - 50': 8, '51 - 54': 9}
    df['dob'] = df['dob'].map(age_mapping)
    bank_mapping = {'2': 1, '8': 2, '9': 3, '11': 4, '13': 5, '14': 6, '22': 7}
    df['bank_id'] = df['bank_id'].map(bank_mapping)
    city_mapping = {'Bandung': 1, 'Jakarta': 2, 'Medan': 3, 'Semarang': 4, 'Surabaya': 5}
    df['city'] = df['city'].map(city_mapping)
    return df

def run_experiment(k, random_state, X, df, experiment_name, output_dir):
    mlflow.set_experiment(experiment_name)  # Set the experiment name
    with mlflow.start_run():
        mlflow.log_param("n_clusters", k)
        mlflow.log_param("random_state", random_state)
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        inertia = kmeans.inertia_
        mlflow.log_metric("inertia", inertia)
        mlflow.sklearn.log_model(kmeans, f"kmeans_model_k{k}")
        centroids = kmeans.cluster_centers_.tolist()
        centroids_path = os.path.join(output_dir, f"kmeans_centroids_k{k}.json")
        with open(centroids_path, "w") as f:
            json.dump({"centroids": centroids}, f)
        mlflow.log_artifact(centroids_path)
        df["cluster"] = kmeans.labels_
        output_path = os.path.join(output_dir, f"clustered_data_k{k}.csv")
        df.to_csv(output_path, index=False)
        mlflow.log_artifact(output_path)
        print(f"K-Means with k={k} completed.")
        print(f"Inertia: {inertia}")
        print(f"Cluster centroids saved to '{centroids_path}'.")
        print(f"Clustered data saved to '{output_path}'.")

def main(k1, k2, random_state, experiment_name):
    output_dir = "./mnt"  # Directory with write permissions

    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    data_path = "Data/data_pii_pst-customer.csv"
    df = pd.read_csv(data_path)
    df = update_colnames(df)
    df_processed = preprocess_data(df)
    df_encoded = encode_data(df_processed)
    X = df_encoded[['dob', 'bank_id', 'city', 'total_transaction', 'income']]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    for k in range(k1, k2 + 1):
        run_experiment(k, random_state, X_scaled, df, experiment_name, output_dir)

if __name__ == "__main__":
    main(1, 16, 42, "KMeans_Clustering_Experiment")
