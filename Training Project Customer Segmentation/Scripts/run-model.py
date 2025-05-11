import pandas as pd
import numpy as np
import json
import joblib
# import cdsw

from joblib import dump
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


# Load the model and scaler
scaler_filename = 'scaler.save'
scaler = joblib.load(scaler_filename)
model_filename = 'pii-customer_kmeans_model.joblib'
model = joblib.load(model_filename)

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

def pred_cust_cluster(args):
  dob = args['dob']
  bank_id = args['bank_id']
  city = args['city']
  total_transaction = args['total_transaction']
  income = args['income']
  
  dob = dob_encode(dob)
  bank_id = bank_encode(bank_id)
  city = city_encode(city)
  features = [dob, bank_id, city, \
            total_transaction, income]
  scaled_features = scaler \
      .transform(np.array(features) \
      .reshape(1, -1))
  result = model.predict(scaled_features)
  return {"cluster_customer": result.item()}

# Example input:
#```
# {
#  "dob": '19 - 22', 
#  "bank_id": '022', 
#  "city": 'Jakarta', 
#  "total_transaction": 731064773, 
#  "income": 7652685 
# }
#```

# Example output:
#```
#{"customer_cluster": 7}
#```


# import pandas as pd
# import numpy as np
# import json
# import joblib
# import cdsw

# from joblib import dump
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import MinMaxScaler


# # Load the model and scaler
# scaler_filename = 'scaler.save'
# scaler = joblib.load(scaler_filename)
# model_filename = 'pii-customer_kmeans_model.joblib'
# model = joblib.load(model_filename)

# ['dob', 'bank_id', 'city', \
#             'total_transaction', 'income']

# def get_dob_cluster(age):
#   age_ranges = [(17, 23, 1), (23, 27, 2), (27, 31, 3), (31, 35, 4), \
#                 (35, 39, 5), (39, 43, 6), (43, 47, 7), (47, 51, 8), \
#                 (51, 100, 9)]
#   for lower, upper, dob in age_ranges:
#     if lower <= age < upper:
#       return dob
#   return 0  # Return 0 for ages outside defined ranges

# # Example usage
# #age = 29
# #dob = get_dob_cluster(age)
# #print(dob)  # Output: 3

# def encode_bank()


# # Main Function
# def pred_cust_cluster(args):  
#   age = args['dob']
#   bank_id = args['bank_id']
#   city = args['city']
#   total_transaction = np.float64(args['total_transaction'])
#   income = np.float64(args['income'])
  
#   # Encode Age --> Age Interval Code
#   dob = get_dob_cluster(age)
#   # Encode Bank ID
#   city_mapping = {'Bandung': 1, 'Jakarta': 2, 'Medan': 3, 'Semarang': 4, 'Surabaya': 5}

#   # Single input value
#   input_city = 'Jakarta'

#   # Fast mapping using the dictionary
#   mapped_value = city_mapping.get(input_city, "Unknown")  # 'Unknown' is a fallback for unmapped cities

#   print(mapped_value)  # Output: 2

  
#   # Encode City
  
  
#   # Track inputs
#   cdsw.track_metric('input_data', data)
  
#   # Predict Data
#   data = [dob, bank_id, city, total_transaction, income]
#   scaled_data = scaler.transform(np.array(data).reshape(1, -1))
#   result = model.predict(scaled_data)
#   return {"cluster_customer": result.item()}
  
# #pred_cust_cluster({ \
# #  "dob": 1, \
# #  "bank_id": 7, \
# #  "city": 2, \
# #  "total_transaction": 731064773, \
# #  "income": 7652685 \
# #})

# # Example input:
# #```
# #{
# #  'age': '19 - 22', 
# #  'bank_id': '022', 
# #  'city': 'Jakarta', 
# #  'total_transaction': 731064773, 
# #  'income': 7652685
# #}
# #```

# # Example output:
# #```
# #{"customer_cluster": 7}
# #```


# #klasifikasi label diketahui
 
# #clustering label tdk diketahui
# from pyspark.sql import SparkSession

# spark = SparkSession \
#         .builder \
#         .appName("SparkHive") \
#         .enableHiveSupport() \
#         .getOrCreate()

# # Create Hive External table

# # IMPORTANT
# # Change pandas dataframe to Spark Dataframe
# pd.DataFrame.iteritems = pd.DataFrame.items
# sparkDF=spark.createDataFrame(df)



# sparkDF.write.mode('overwrite').saveAsTable("demo_cust_segmentation.customer_segmentation")  

# sparkDF.createOrReplaceTempView('mytempTable')

# spark.sql("""SELECT * FROM mytempTable LIMIT 5""").show()

# sparkDF.write.saveAsTable("demo_cust_segmentation.testing2")

# #  .option("path", "/path/to/external/table") \  
  
  

  
# import sys
# print(sys.version)

# import pandas as pd
# from pyspark.sql import *
# from pyspark.sql.functions import *


# spark = SparkSession\
#   .builder\
#   .appName("Query CML")\
#   .config("spark.dynamicAllocation.enabled", "false")\
#   .config("spark.executor.instances", "1")\
#   .config("spark.executor.cores", "1")\
#   .config("spark.executor.memory", "1g")\
#   .config("spark.yarn.executor.memoryOverhead", "1g")\
#   .config("sparl.sql.parquet.compression.codec", "snappy")\
#   .config("spark.shuffle.service.enabled","false")\
#   .config("hive.metastore.uris","thrift://cdp.wleowleo.uk:9083")\
#   .config("hive.metastore.kerberos.principal","hive/cdp.wleowleo.uk@WLEOWLEO.UK")\
#   .config("spark.yarn.principal","hive/cdp.wleowleo.uk@WLEOWLEO.UK")\
#   .config("spark.yarn.keytab","hive.keytab")\
#   .config("spark.sql.hive.hiveserver2.jdbc.url", "jdbc:hive2://cdp.wleowleo.uk:2181/default;httpPath=cliservice;serviceDiscoveryMode=zooKeeper;transportMode=http;zooKeeperNamespace=hiveserver2")\
#   .config("spark.sql.hive.hiveserver2.jdbc.url.principal", "hive/cdp.wleowleo.uk@WLEOWLEO.UK")\
#   .enableHiveSupport()\
#   .getOrCreate()

# sparkDF.createOrReplaceTempView('mytempTable')
# df = spark.sql("SELECT * FROM `demo_cust_segmentation`.`customer`")
# spark.sql("""SELECT * FROM mytempTable LIMIT 5""").show()

# df.show()  
  
  
  
  
  
  

  

# #type(X_scaled)
# #b = np.ndarray([0. , 1. , 0.25      , 0.73075764, 0.70657877])
# #a = scaler.transform([0. , 1. , 0.25      , 0.73075764, 0.70657877])
# #z = df_encoded[:1][['dob', 'bank_id', 'city', 'total_transaction', 'income']]
# #a = scaler.transform(z)
# #a

# #  return {"customer_cluster": int(7)}
# #  return {"customer_cluster": np.float64(total_transaction)}
  
  
  
  
#   # Encoding 'dob' (age interval) into 9 clusters
#   age_mapping = {
#       '19 - 22': 1, '23 - 26': 2, '27 - 30': 3, '31 - 34': 4, '35 - 38': 5,
#       '39 - 42': 6, '43 - 46': 7, '47 - 50': 8, '51 - 54': 9
#   }
#   # Encoding 'bank_id'
#   bank_mapping = {'002': 1, '008': 2, '009': 3, '011': 4, '013': 5, '014': 6, '022': 7}
#   # Encoding 'city'
#   city_mapping = {'Bandung': 1, 'Jakarta': 2, 'Medan': 3, 'Semarang': 4, 'Surabaya': 5}
