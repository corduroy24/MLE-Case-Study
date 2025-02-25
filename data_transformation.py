from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from pyspark.sql.functions import year, month, dayofmonth, dayofweek, datediff
from pyspark.sql.functions import col
import pandas as pd

def feature_engineering_spark(df):    
    # date-based features
    df = df.withColumn("search_year", year(col("searchDate")))
    df = df.withColumn("search_month", month(col("searchDate")))
    df = df.withColumn("search_weekday", dayofweek(col("searchDate")))
    df = df.withColumn("search_day", dayofmonth(col("searchDate")))
    df = df.withColumn("stayDuration", datediff(col("checkOutDate"), col("checkInDate")))
    df = df.withColumn("days_until_checkin", datediff(col("checkinDate"), col("searchDate")))

    return df


def drop_columns(df):
    # remove unneeded columns
    columns_to_drop = ['userId', 'searchId', 'checkinDate', 'searchDate', 'checkOutDate', 
                       'hotelId', 'brandId', 'missing_count', 'most_common_destination', 
                       'clickLabel']
    
    df = df.drop(*columns_to_drop)

    return df
    
def balance_target_var(X, y):
    # upsample only 30%
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    return X_resampled, y_resampled


def feature_engineering_pandas(df):
    df_pandas = df.toPandas()

    le = LabelEncoder()
    df_pandas['destinationName'] = le.fit_transform(df_pandas['destinationName'])

    rel_cols = ['vipTier', 'deviceCode']
    df_pandas = pd.get_dummies(df_pandas, columns=rel_cols, drop_first=True)  

    return df_pandas