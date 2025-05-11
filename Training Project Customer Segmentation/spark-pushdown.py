import sys
print(sys.version)

import pandas as pd
from pyspark.sql import *
from pyspark.sql.functions import *


spark = SparkSession\
  .builder\
  .appName("Query CML")\
  .config("spark.dynamicAllocation.enabled", "false")\
  .config("spark.executor.instances", "1")\
  .config("spark.executor.cores", "1")\
  .config("spark.executor.memory", "1g")\
  .config("spark.yarn.executor.memoryOverhead", "1g")\
  .config("sparl.sql.parquet.compression.codec", "snappy")\
  .config("spark.shuffle.service.enabled","false")\
  .config("hive.metastore.uris","thrift://cdp.wleowleo.uk:9083")\
  .config("hive.metastore.kerberos.principal","hive/cdp.wleowleo.uk@WLEOWLEO.UK")\
  .config("spark.yarn.principal","hive/cdp.wleowleo.uk@WLEOWLEO.UK")\
  .config("spark.yarn.keytab","hive.keytab")\
  .config("spark.sql.hive.hiveserver2.jdbc.url", "jdbc:hive2://cdp.wleowleo.uk:2181/default;httpPath=cliservice;serviceDiscoveryMode=zooKeeper;transportMode=http;zooKeeperNamespace=hiveserver2")\
  .config("spark.sql.hive.hiveserver2.jdbc.url.principal", "hive/cdp.wleowleo.uk@WLEOWLEO.UK")\
  .enableHiveSupport()\
  .getOrCreate()

df = spark.sql("SELECT * FROM datalake_pst.customer limit 10")


df.show()
