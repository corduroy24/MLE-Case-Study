# Import relevant libaries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from data_ingestion import init_spark, load_data, preprocessing
from data_transformation import feature_engineering_spark, feature_engineering_pandas, balance_target_var, drop_columns
from train_classifier import train_classifier

def main():
    print("Loading Data...")
    spark = init_spark()
    df = load_data(spark)
    df = preprocessing(df)
    print(df.show(5))


    print("Feature Engineering...")
    df = feature_engineering_spark(df)
    df = drop_columns(df)
    df_pandas = feature_engineering_pandas(df)

    print('df_pandas type', type(df_pandas))

    X = df_pandas.drop(columns=["bookingLabel"])
    y = df_pandas["bookingLabel"]

    X_resampled, y_resampled = balance_target_var(X, y)

    print(X_resampled.head())

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    print("Training Model...")
    model = train_classifier(X_train, y_train, X_test, y_test)

    print("Pipeline Completed")

if __name__ == "__main__":
    main()

