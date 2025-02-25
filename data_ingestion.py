from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, countDistinct
from pyspark.sql.functions import year, month, dayofmonth, dayofweek, datediff
from config import DATA_PATH


def init_spark():
    spark = SparkSession.builder.appName("HotelBooking_Probability_Prediction_Pipeline").getOrCreate()
    return spark

def load_data(spark):
    df = spark.read.parquet(DATA_PATH)
    return df

def preprocessing(df):
    # change data type of date columns
    df = df.withColumn("searchDate", col("searchDate").cast("date"))
    df = df.withColumn("checkinDate", col("checkinDate").cast("date"))
    df = df.withColumn("checkOutDate", col("checkOutDate").cast("date"))
    
    df = df.withColumn(
        "destinationName",
        when(col("destinationName").isNull(), "Unknown").otherwise(col("destinationName"))
    )
    return df

