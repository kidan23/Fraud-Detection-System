import pandas as pd

from src.preprocessor import (
    convert_ip_to_int,
    ensure_ip_bounds_int,
    fix_datetime_columns,
    drop_duplicates,
    merge_ip_country,
)


def test_convert_ip_to_int():
    df = pd.DataFrame({'ip_address': ['127.0.0.1', '8.8.8.8', 'invalid']})
    out = convert_ip_to_int(df, 'ip_address', 'ip_address_int')
    assert out['ip_address_int'].iloc[0] > 0
    assert out['ip_address_int'].iloc[1] > 0
    assert out['ip_address_int'].iloc[2] == 0


def test_fix_datetime_columns():
    df = pd.DataFrame({
        'signup_time': ['2020-01-01', '2020-01-02'],
        'purchase_time': ['2020-01-02 12:00:00', '2020-01-03 15:00:00']
    })
    out = fix_datetime_columns(df, ['signup_time', 'purchase_time'])
    assert pd.api.types.is_datetime64_any_dtype(out['signup_time'])
    assert pd.api.types.is_datetime64_any_dtype(out['purchase_time'])


def test_drop_duplicates():
    df = pd.DataFrame({'a': [1, 1, 2]})
    out = drop_duplicates(df)
    assert out.shape[0] == 2


def test_merge_ip_country():
    fraud = pd.DataFrame({
        'user_id': [1, 2],
        'ip_address_int': [10, 30],
        'ip_address': ['0.0.0.10', '0.0.0.30'],
        'class': [0, 1]
    })

    ip_country = pd.DataFrame({
        'lower_bound_ip_address': [0, 20],
        'upper_bound_ip_address': [19, 40],
        'country': ['A', 'B']
    })

    merged = merge_ip_country(fraud, ip_country,
                              left_ip_col='ip_address_int',
                              right_lower='lower_bound_ip_address',
                              right_upper='upper_bound_ip_address')
    assert 'country' in merged.columns
    assert merged.shape[0] == 2
