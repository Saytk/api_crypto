"""
Chargement BigQuery (données 1 minute) + agrégation horaire (optionnelle).
"""

import pandas as pd, yaml, os
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.oauth2 import service_account
from pathlib import Path
from crypto_signals.src.utils.logger import get_logger

log = get_logger()

CFG = yaml.safe_load(open(Path(__file__).parents[1] / "config" / "config.yaml"))
CREDENTIALS_PATH = Path(__file__).parents[1] / "config" / "credentials.json"

def _client():
    """Create and return a BigQuery client. The caller is responsible for closing the client."""
    if os.path.exists(CREDENTIALS_PATH):
        credentials = service_account.Credentials.from_service_account_file(
            CREDENTIALS_PATH,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        return bigquery.Client(project=CFG["project"], credentials=credentials)
    else:
        log.warning("Credentials file not found. Using default credentials.")
        return bigquery.Client(project=CFG["project"])

def load_minute(symbol: str = "BTCUSDT",
                days: int = 7,
                max_rows: int = 100_000) -> pd.DataFrame:
    start = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    table = f"{CFG['project']}.{CFG['table_minute']}"
    query = f"""
        SELECT timestamp_utc, open, high, low, close,
               volume, quote_volume, nb_trades
        FROM `{table}`
        WHERE symbol = '{symbol}'
          AND timestamp_utc >= TIMESTAMP('{start}')
        ORDER BY timestamp_utc
        LIMIT {max_rows}
    """
    client = _client()
    try:
        df = client.query(query).to_dataframe()
        log.info(f"{symbol} – rows loaded: {len(df)}")
        return df
    finally:
        client.close()
        log.debug(f"BigQuery client closed after loading {symbol} data")

def minute_to_hour(df_min: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "open":  "first",
        "high":  "max",
        "low":   "min",
        "close": "last",
        "volume":       "sum",
        "quote_volume": "sum",
        "nb_trades":    "sum"
    }
    df_h = (df_min
            .set_index("timestamp_utc")
            .resample("1H")
            .agg(agg)
            .dropna()
            .reset_index())
    return df_h

def load_last_candle(symbol: str = "BTCUSDT") -> dict:
    """
    Retrieve only the last (most recent) candle for a given symbol.

    Args:
        symbol: Trading pair symbol (e.g., BTCUSDT, ETHUSDT)

    Returns:
        A dictionary containing the last candle data
    """
    table = f"{CFG['project']}.{CFG['table_minute']}"
    query = f"""
        SELECT timestamp_utc, open, high, low, close,
               volume, quote_volume, nb_trades
        FROM `{table}`
        WHERE symbol = '{symbol}'
        ORDER BY timestamp_utc DESC
        LIMIT 1
    """
    client = _client()
    try:
        df = client.query(query).to_dataframe()
        if df.empty:
            log.warning(f"No data found for {symbol}")
            return {}

        # Convert to dictionary and format timestamp
        candle = df.iloc[0].to_dict()
        candle["timestamp_utc"] = candle["timestamp_utc"].isoformat()

        log.info(f"{symbol} – last candle loaded: {candle['timestamp_utc']}")
        return candle
    finally:
        client.close()
        log.debug(f"BigQuery client closed after loading last candle for {symbol}")
