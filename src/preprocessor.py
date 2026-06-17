from typing import Optional

import ipaddress
import pandas as pd


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
	try:
		return df.drop_duplicates()
	except Exception as exc:
		raise RuntimeError(f"Failed to drop duplicates: {exc}") from exc


def fix_datetime_columns(df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
	if not columns:
		return df
	df = df.copy()
	try:
		for c in columns:
			df[c] = pd.to_datetime(df[c])
		return df
	except Exception as exc:
		raise RuntimeError(f"Failed to convert datetime columns: {exc}") from exc


def convert_ip_to_int(df: pd.DataFrame, ip_col: str = "ip_address", out_col: str = "ip_address_int") -> pd.DataFrame:
	df = df.copy()

	def _to_int(ip):
		try:
			return int(ipaddress.ip_address(int(ip))) if str(ip).isdigit() else int(ipaddress.ip_address(str(ip)))
		except Exception:
			return 0

	try:
		df[out_col] = df[ip_col].apply(_to_int)
		return df
	except Exception as exc:
		raise RuntimeError(f"Failed to convert IP addresses: {exc}") from exc


def ensure_ip_bounds_int(df: pd.DataFrame, lower_col: str = "lower_bound_ip_address", upper_col: str = "upper_bound_ip_address") -> pd.DataFrame:
	df = df.copy()
	try:
		df[lower_col] = df[lower_col].astype(int)
		df[upper_col] = df[upper_col].astype(int)
		return df
	except Exception as exc:
		raise RuntimeError(f"Failed to coerce IP bound columns to int: {exc}") from exc


def merge_ip_country(fraud_df: pd.DataFrame,
					 ip_country_df: pd.DataFrame,
					 left_ip_col: str = "ip_address_int",
					 right_lower: str = "lower_bound_ip_address",
					 right_upper: str = "upper_bound_ip_address") -> pd.DataFrame:
	
	try:
		ip_country_sorted = ip_country_df.sort_values(right_lower).reset_index(drop=True)
		fraud_sorted = fraud_df.sort_values(left_ip_col).reset_index(drop=True)

		merged = pd.merge_asof(
			fraud_sorted,
			ip_country_sorted,
			left_on=left_ip_col,
			right_on=right_lower,
			direction='backward'
		)

		merged = merged[merged[left_ip_col] <= merged[right_upper]]
		return merged
	except Exception as exc:
		raise RuntimeError(f"Failed to merge IP country data: {exc}") from exc

