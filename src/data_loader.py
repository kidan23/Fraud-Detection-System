from pathlib import Path
from typing import Tuple

import pandas as pd


def load_datasets(folder: str = "data/raw") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	p = Path(folder)
	try:
		fraud_df = pd.read_csv(p / "Fraud_Data.csv")
	except Exception as exc:
		raise RuntimeError(f"Failed to load Fraud_Data.csv: {exc}") from exc

	try:
		ip_country_df = pd.read_csv(p / "IpAddress_to_Country.csv")
	except Exception as exc:
		raise RuntimeError(f"Failed to load IpAddress_to_Country.csv: {exc}") from exc

	try:
		credit_df = pd.read_csv(p / "creditcard.csv")
	except Exception as exc:
		raise RuntimeError(f"Failed to load creditcard.csv: {exc}") from exc

	return fraud_df, ip_country_df, credit_df

