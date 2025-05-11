CREATE TABLE testing_wawan.mall_customers (
    CustomerID INT,
    Gender STRING,
    Age INT,
    `Annual Income (k$)` INT,
    `Spending Score (1-100)` INT
) ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' STORED AS TEXTFILE;