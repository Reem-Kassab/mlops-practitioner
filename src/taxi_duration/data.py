import pandas as pd


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Loads raw NYC taxi parquet data, calculates duration, 
    and applies basic sanity cleaning.
    """
    # 1. Reading the data
    df = pd.read_parquet(file_path)
    # 2. calculate the duration
    df['duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    df = df[(df["trip_distance"] >= 0.1) & (df["duration"] >= 0.1)]
    # droping the leackage columns
    columns_to_drop = [
        'fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount',
        'ehail_fee', 'improvement_surcharge', 'total_amount', 
        'payment_type', 'congestion_surcharge', 'cbd_congestion_fee'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    return df