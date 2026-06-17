from typing import Optional

import pandas as pd


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek
        df['day_of_month'] = df['purchase_time'].dt.day
        return df
    except Exception as exc:
        raise RuntimeError(f"Failed to create time features: {exc}") from exc


def create_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    try:
        user_transaction_count = df.groupby('user_id').size().rename('transaction_count')
        df['transaction_count'] = df['user_id'].map(user_transaction_count)
        df['transaction_velocity'] = df['transaction_count'] / (df['time_since_signup'] + 1)
        df['is_new_account'] = (df['time_since_signup'] < 24).astype(int)
        return df
    except Exception as exc:
        raise RuntimeError(f"Failed to create velocity features: {exc}") from exc
