import pandas as pd

from src.feature_engineer import create_time_features, create_velocity_features


def test_create_time_features():
    df = pd.DataFrame({
        'signup_time': pd.to_datetime(['2020-01-01 00:00:00', '2020-01-02 00:00:00']),
        'purchase_time': pd.to_datetime(['2020-01-01 12:00:00', '2020-01-03 00:00:00'])
    })
    out = create_time_features(df)
    assert 'time_since_signup' in out.columns
    assert 'hour_of_day' in out.columns
    assert out['time_since_signup'].iloc[0] == 12


def test_create_velocity_features():
    df = pd.DataFrame({
        'user_id': [1, 1, 2],
        'time_since_signup': [1, 2, 10],
        'class': [0, 1, 0]
    })
    out = create_velocity_features(df)
    assert 'transaction_count' in out.columns
    assert out.loc[out['user_id'] == 1, 'transaction_count'].iloc[0] == 2
    assert 'transaction_velocity' in out.columns
    assert 'is_new_account' in out.columns
