import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# Create a SparkSession
spark = SparkSession.builder.appName("ReadHiveTable").getOrCreate()

# Read data from the Hive table
df = spark.sql("SELECT * FROM testing_wawan.mall_customers")

spark.sql("SHOW DATABASES")

# Show the DataFrame
df.show(5)

import pyspark
from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder.appName("ReadHiveTable").getOrCreate()

# Assuming the table data is located at /user/hive/warehouse/testing_wawan.db/mall_customers
hdfs_path = "hdfs://cdp.wleowleo.uk:8020/warehouse/tablespace/managed/hive/testing_wawan.db/mall_customers"

schema = StructType([
    StructField("CustomerID", IntegerType(), False),
    StructField("Gender", StringType(), False),
    StructField("Age", IntegerType(), False),
    StructField("AnnualIncome", IntegerType(), False),  # Rename the column
    StructField("SpendingScore", IntegerType(), False)  # Rename the column
])

# Load data using the HDFS path
df = spark.read.format("parquet").schema(schema).load(hdfs_path)

# Rename the columns back to their original names (optional)
df = df.withColumnRenamed("AnnualIncome", "Annual Income (k$)").withColumnRenamed("SpendingScore", "Spending Score (1-100)")

# Show the DataFrame
df.limit(5).show()