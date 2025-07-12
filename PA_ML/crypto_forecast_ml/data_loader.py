# crypto_forecast_ml/data_loader.py

import os
import pandas as pd
from google.cloud import bigquery
from datetime import datetime, timedelta
from pathlib import Path
import logging

# ⚙️ Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_crypto_data(symbol: str = "BTCUSDT", days: int = 7, max_rows: int = 100_000) -> pd.DataFrame:
    logger.info("🟡 load_crypto_data() CALLED")

    # 🔧 Recherche le chemin absolu vers le dossier crypto_forecast_ml
    for p in Path(__file__).resolve().parents:
        if (p / "crypto_forecast_ml" / "config" / "credentials.json").exists():
            credentials_path = p / "crypto_forecast_ml" / "config" / "credentials.json"
            break
    else:
        raise FileNotFoundError("❌ Impossible de localiser crypto_forecast_ml/config/credentials.json")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
    logger.info(f"✅ Using credentials: {credentials_path}")

    # BigQuery client
    client = bigquery.Client()

    # Fenêtre temporelle
    start_date = (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%d')
    query = f"""
        SELECT timestamp_utc, open, high, low, close, volume, quote_volume, nb_trades
        FROM `feisty-coder-461708-m9.data_bronze.RAW_CRYPTO_KLINES_1MIN`
        WHERE symbol = '{symbol}'
          AND timestamp_utc >= TIMESTAMP('{start_date}')
        ORDER BY timestamp_utc ASC
        LIMIT {max_rows}
    """

    logger.info(f"📥 Launching BigQuery query for {symbol}...")
    df = client.query(query).to_dataframe()
    logger.info(f"📊 Loaded {len(df)} rows.")

    return df


def load_crypto_data_custom_range(symbol: str, start_date: str, end_date: str, max_rows: int = 100_000) -> pd.DataFrame:
    logger.info(f"🟡 load_crypto_data_custom_range() CALLED with symbol={symbol}, start={start_date}, end={end_date}")

    # 🔍 Recherche les credentials
    for p in Path(__file__).resolve().parents:
        if (p / "crypto_forecast_ml" / "config" / "credentials.json").exists():
            credentials_path = p / "crypto_forecast_ml" / "config" / "credentials.json"
            break
    else:
        raise FileNotFoundError("❌ Impossible de localiser crypto_forecast_ml/config/credentials.json")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
    logger.info(f"✅ Using credentials: {credentials_path}")

    # BigQuery client
    client = bigquery.Client()

    query = f"""
        SELECT timestamp_utc, open, high, low, close, volume, quote_volume, nb_trades
        FROM `feisty-coder-461708-m9.data_bronze.RAW_CRYPTO_KLINES_1MIN`
        WHERE symbol = '{symbol}'
          AND timestamp_utc >= TIMESTAMP('{start_date}')
          AND timestamp_utc <= TIMESTAMP('{end_date}')
        ORDER BY timestamp_utc ASC
        LIMIT {max_rows}
    """

    logger.info(f"📥 Running query for range: {start_date} to {end_date}")
    logger.info(f"📥 Running query : {query}")
    df = client.query(query).to_dataframe()
    logger.info(f"📊 Loaded {len(df)} rows.")
    return df
