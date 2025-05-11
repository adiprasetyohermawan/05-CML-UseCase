import pyspark
import pyspark.pandas as ps
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cdsw

from joblib import dump
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from pyspark.sql import SparkSession


# Create a SparkSession
spark = SparkSession \
        .builder \
        .appName("SparkHive") \
        .enableHiveSupport() \
        .getOrCreate()

# Check connection & Show all databases
spark.sql("SHOW DATABASES").show()

# Database: demo_cust_segmentation
# Table: superstore_datasets_pst
# 1. Read data from the Hive table
df = spark \
      .sql("SELECT * FROM `demo_cust_segmentation`.`customer`") \
      .toPandas()
#df = df.toPandas()

# Show the DataFrame
df.columns
df.head(5)
df.info()

# =============================================================

# 2. Data Preprocessing
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame by drop unused columns, converting column data types, and encoding columns.
    Parameters:
    - df (pd.DataFrame): Input DataFrame containing various data types.
    Returns:
    - pd.DataFrame: The processed DataFrame with specified columns converted to appropriate data types.
    """

    # Drop unused columns for analysis
    df = df.drop(columns=['email', \
                        'phone_numb', \
                        'account_number', \
                        'ktp', \
                        'postal_code', \
                        'address'])

    # Convert specified columns to string type
    string_columns = ['cust_id', 'name', 'bank_id', 'dob', 'city']
    df[string_columns] = df[string_columns].astype('string')
    # Convert specified columns to float64 (decimal)
    decimal_columns = ['total_transaction', 'income']
    df[decimal_columns] = df[decimal_columns].astype('float64')

    return df

# Example usage:
df_processed = preprocess_data(df)
df_processed.info()
df_processed.head(5)

==========================================================================
# Encode Data

def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    # Encoding 'dob' (age interval) into 9 clusters
    age_mapping = {
        '19 - 22': 1, '23 - 26': 2, '27 - 30': 3, '31 - 34': 4, '35 - 38': 5,
        '39 - 42': 6, '43 - 46': 7, '47 - 50': 8, '51 - 54': 9
    }
    df['dob'] = df['dob'].map(age_mapping)
    # Encoding 'bank_id'
    bank_mapping = {'002': 1, '008': 2, '009': 3, '011': 4, '013': 5, '014': 6, '022': 7}
    df['bank_id'] = df['bank_id'].map(bank_mapping)
    # Encoding 'city'
    city_mapping = {'Bandung': 1, 'Jakarta': 2, 'Medan': 3, 'Semarang': 4, 'Surabaya': 5}
    df['city'] = df['city'].map(city_mapping)

    return df

df_encoded = encode_data(df_processed)
df_encoded.head(5)
df_encoded.info()
  
==============================================================================
  
# 3. Feature Scaling
X = df_encoded[['dob', 'bank_id', 'city', 'total_transaction', 'income']]
scaler = MinMaxScaler()
#X_scaled = scaler.fit_transform(X)
X_scaled = scaler.fit(X)
X_scaled = scaler.transform(X)
X_scaled

# 4. K-Means Clustering
# Based on the Elbow method, choose the optimal K
optimal_k = 8
# Apply K-Means with the optimal K
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
clusters

# 5. Attach the cluster labels to the original dataframe
df_encoded['cluster'] = clusters
df_encoded.head(5)

# Output
# Save Scaler Training Data
filename = 'scaler.save'
dump(scaler, filename)
cdsw.track_file(scaler)

# Save the KMeans model
filename = 'pii-customer_kmeans_model.joblib'
dump(kmeans, filename)
cdsw.track_file(filename)

#Save files needed
#df_encoded.to_csv('df_encoded.csv', index=False)
#df_processed.to_csv('df_processed.csv', index=False)
#df_result.csv to save in hive or impala, consist of df_processed + cluster column


import pyspark
import pyspark.pandas as ps
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cdsw
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from pyspark.sql import SparkSession


def create_spark_session():
    """Create a Spark session."""
    spark = SparkSession \
        .builder \
        .appName("SparkHive") \
        .enableHiveSupport() \
        .getOrCreate()
    return spark


def read_data(spark: SparkSession) -> pd.DataFrame:
    """Read data from Hive and return as a Pandas DataFrame."""
    df = spark \
        .sql("SELECT * FROM `demo_cust_segmentation`.`customer`") \
        .toPandas()
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DataFrame by dropping unused columns and converting column data types."""
    # Drop unused columns
    df = df.drop(columns=['email', 'phone_numb', 'account_number', 'ktp', 'postal_code', 'address'])

    # Convert specified columns to string type
    string_columns = ['cust_id', 'name', 'bank_id', 'dob', 'city']
    df[string_columns] = df[string_columns].astype('string')

    # Convert specified columns to float64 (decimal)
    decimal_columns = ['total_transaction', 'income']
    df[decimal_columns] = df[decimal_columns].astype('float64')

    return df


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns like 'dob', 'bank_id', and 'city'."""
    # Encoding 'dob' (age interval) into 9 clusters
    age_mapping = {
        '19 - 22': 1, '23 - 26': 2, '27 - 30': 3, '31 - 34': 4, '35 - 38': 5,
        '39 - 42': 6, '43 - 46': 7, '47 - 50': 8, '51 - 54': 9
    }
    df['dob'] = df['dob'].map(age_mapping)

    # Encoding 'bank_id'
    bank_mapping = {'002': 1, '008': 2, '009': 3, '011': 4, '013': 5, '014': 6, '022': 7}
    df['bank_id'] = df['bank_id'].map(bank_mapping)

    # Encoding 'city'
    city_mapping = {'Bandung': 1, 'Jakarta': 2, 'Medan': 3, 'Semarang': 4, 'Surabaya': 5}
    df['city'] = df['city'].map(city_mapping)

    return df


def scale_features(df: pd.DataFrame) -> np.ndarray:
    """Scale the features using MinMaxScaler."""
    X = df[['dob', 'bank_id', 'city', 'total_transaction', 'income']]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def perform_kmeans(X_scaled: np.ndarray, optimal_k: int = 8) -> np.ndarray:
    """Apply KMeans clustering."""
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters, kmeans


def save_model_and_scaler(kmeans, scaler):
    """Save the KMeans model and the scaler."""
    # Save Scaler
    scaler_filename = 'scaler.save'
    dump(scaler, scaler_filename)
    cdsw.track_file(scaler_filename)

    # Save KMeans Model
    kmeans_filename = 'pii-customer_kmeans_model.joblib'
    dump(kmeans, kmeans_filename)
    cdsw.track_file(kmeans_filename)


def attach_clusters(df: pd.DataFrame, clusters: np.ndarray) -> pd.DataFrame:
    """Attach the cluster labels to the DataFrame."""
    df['cluster'] = clusters
    return df


def main():
    """Main function to execute the steps."""
    # Step 1: Create Spark session and read data
    spark = create_spark_session()
    df = read_data(spark)

    # Step 2: Preprocess the data
    df_processed = preprocess_data(df)

    # Step 3: Encode categorical features
    df_encoded = encode_data(df_processed)

    # Step 4: Scale the features
    X_scaled, scaler = scale_features(df_encoded)

    # Step 5: Perform K-Means Clustering
    optimal_k = 8  # Replace with the optimal K found through the elbow method
    clusters, kmeans = perform_kmeans(X_scaled, optimal_k)

    # Step 6: Attach the clusters to the DataFrame
    df_with_clusters = attach_clusters(df_encoded, clusters)

    # Step 7: Save the model and scaler
    save_model_and_scaler(kmeans, scaler)

    # Output: Display the result
    print(df_with_clusters.head(5))

    # Save the final result if needed
    df_with_clusters.to_csv('df_with_clusters.csv', index=False)
    cdsw.track_file('df_with_clusters.csv')


if __name__ == "__main__":
    main()



# ==================================================================================================
# CODE DI BAWAH YG DIPAKAI
import pyspark
import pyspark.pandas as ps
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cdsw
import joblib
import sys
print(sys.version)

from joblib import dump
from typing import Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from pyspark.sql import SparkSession
from pyspark.sql import *
from pyspark.sql.functions import *

# ================================================================================
# Step 1: Create Spark Session
def create_spark_session(app_name: str) -> SparkSession:
  """
  Initialize a Spark session.
  Parameters:
  - app_name (str): Name of the Spark application.
  Returns:
  - SparkSession: Spark session object with Hive support.
  """
  spark = SparkSession \
          .builder \
          .appName(app_name) \
          .enableHiveSupport() \
          .getOrCreate()
  return spark

app_name = "CustomerSegmentation"
spark = create_spark_session(app_name)
# Check connection & Show all databases
# spark.sql("SHOW DATABASES").show()

# ================================================================================
# Step 2: Load Data from Hive
def load_data_from_hive(spark: SparkSession, database: str, table: str) -> pd.DataFrame:
  """
  Load data from a Hive table.
  Parameters:
  - spark (SparkSession): Spark session with Hive support.
  - database (str): Hive database name.
  - table (str): Hive table name.
  Returns:
  - pd.DataFrame: Data loaded from Hive table as a Pandas DataFrame.
  """
  df = spark \
      .sql(f"SELECT * FROM `{database}`.`{table}`") \
      .toPandas()
  return df

database = "demo_cust_segmentation"
table = "customer"
df = load_data_from_hive(spark, database, table)
print("Data Loaded:")
print(df.head(3))
print(df.columns)
print(df.info())

# ================================================================================
# Step 3: Preprocess Data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
  """
  Preprocess the DataFrame by dropping unused columns and converting data types.
  Parameters:
  - df (pd.DataFrame): Input DataFrame.
  Returns:
  - pd.DataFrame: Processed DataFrame.
  """
  # Drop unused columns
  df = df.drop(columns=['email', \
                      'phone_numb', \
                      'account_number', \
                      'ktp', \
                      'postal_code', \
                      'address'])
  # Convert specified columns to string type
  string_columns = ['cust_id', 'name', 'bank_id', 'dob', 'city']
  df[string_columns] = df[string_columns].astype('string')
  # Convert specified columns to float64 (decimal)
  decimal_columns = ['total_transaction', 'income']
  df[decimal_columns] = df[decimal_columns].astype('float64')
  return df

df_processed = preprocess_data(df)
print("\nData After Preprocessing:")
df_processed.info()
df_processed.head(3)

# ================================================================================
# Step 4: Encode Data
def encode_data(df: pd.DataFrame) -> pd.DataFrame:
  """
    Encode categorical data in the DataFrame.
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    Returns:
    - pd.DataFrame: Encoded DataFrame.
    """
  # Encoding mappings
  age_mapping = {
      '19 - 22': 1, '23 - 26': 2, '27 - 30': 3, \
      '31 - 34': 4, '35 - 38': 5, '39 - 42': 6, \
      '43 - 46': 7, '47 - 50': 8, '51 - 54': 9
  }
  bank_mapping = {'002': 1, '008': 2, '009': 3, \
                  '011': 4, '013': 5, '014': 6, '022': 7}
  city_mapping = {'Bandung': 1, 'Jakarta': 2, 'Medan': 3, \
                  'Semarang': 4, 'Surabaya': 5}
  # Apply mappings
  df['dob'] = df['dob'].map(age_mapping)
  df['bank_id'] = df['bank_id'].map(bank_mapping)
  df['city'] = df['city'].map(city_mapping)
  return df

df_encoded = encode_data(df_processed)
print("\nData After Encoding:")
df_encoded.head(3)
df_encoded.info()
  
# ================================================================================
# Step 5: Scale Features
def scale_features(df: pd.DataFrame, \
                   scaler, \
                   filename: str, \
                   features: list) -> np.ndarray:
  """
  Scale selected features using selected scaler.
  Parameters:
  - df (pd.DataFrame): Input DataFrame.
  - scaler: Scaler instance (e.g., MinMaxScaler()).
  - features (list): List of features to scale.
  Returns:
  - np.ndarray: Scaled features array.
  """
  scaler = MinMaxScaler()
  scaled_features = scaler.fit_transform(df[features])
  # Save the scaler for future use
  dump(scaler, filename)
  cdsw.track_file(scaler)
  return scaled_features

scaler = MinMaxScaler()
filename = 'scaler.save'
features = ['dob', 'bank_id', 'city', \
            'total_transaction', 'income']
X_scaled = scale_features(df_encoded, scaler, filename, features)
print("Scaled Features:\n", X_scaled[:5])

# ================================================================================
# Step 6: Apply KMeans Clustering
def apply_kmeans(scaled_data: np.ndarray, \
                 optimal_k: int, \
                 filename: str) -> np.ndarray:
  """
  Apply K-Means clustering.
  Parameters:
  - scaled_data (np.ndarray): Scaled data for clustering.
  - optimal_k (int): Optimal number of clusters.
  Returns:
  - np.ndarray: Cluster labels.
  """
  kmeans = KMeans(n_clusters=optimal_k, \
                  init='k-means++', \
                  max_iter=300, \
                  n_init=10, \
                  random_state=42)
  clusters = kmeans.fit_predict(scaled_data)
  # Save the KMeans model
  dump(kmeans, filename)
  cdsw.track_file(filename)
  return clusters

# Replace with the optimal number of clusters as determined 
# by the elbow method or other techniques
optimal_k = 4
filename = 'pii-customer_kmeans_model.joblib'
clusters = apply_kmeans(X_scaled, optimal_k, filename)
print("\nCluster Labels:")
print(clusters[:5])

# ================================================================================
# Step 7: Save Results with Cluster Labels
def save_results(df: pd.DataFrame, \
                 clusters: np.ndarray, \
                 database: str, \
                 table: str
                ) -> Tuple[pd.DataFrame, \
                     pyspark.sql.dataframe.DataFrame]:
  """
  Save DataFrame with cluster labels.
  Parameters:
  - df (pd.DataFrame): Original DataFrame.
  - clusters (np.ndarray): Cluster labels.
  - database (str): Hive database name.
  - table (str): Hive table name.
  """
  df['cluster'] = clusters
  # Save as CSV
  df.to_csv('df_result.csv', index=False)
  
  # Change pandas dataframe to Spark Dataframe
  pd.DataFrame.iteritems = pd.DataFrame.items
  df_spark=spark.createDataFrame(df)
  df_spark.createOrReplaceTempView('mytempTable')
  # Save df result into external table in Hive
  df_spark.write.saveAsTable(f"{database}.{table}")
  
  return df, df_spark

database = "demo_cust_segmentation"
table = "result_customer_segmentation"
df_result, sparkDF = save_results(df, clusters, database, table)
print("\nResults Saved with Cluster Labels.")
print("Check the 'df_result.csv' file and result table in Hive.")
print(df_result.head(3))
print(sparkDF.show(3))