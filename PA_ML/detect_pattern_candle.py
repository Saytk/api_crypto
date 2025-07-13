"""
label_candles_scipy.py

Charge les données de crypto minute via load_crypto_data_custom_range
et applique K-Means (via scipy) pour classifier chaque bougie.

Ajoute une colonne 'candle_type' (cluster ID) au DataFrame et exporte en CSV.
"""

import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans2, whiten

from crypto_forecast_ml.data_loader import load_crypto_data_custom_range

# === Paramètres ===
SYMBOL = "BTCUSDT"
START_DATE = '2025-06-01T09:00:00'
END_DATE = "2025-07-13T09:00:00"
N_CLUSTERS = 10
OUTPUT_FILE = f"{SYMBOL}_labeled_candles.csv"

# === Chargement des données ===
df = load_crypto_data_custom_range(symbol=SYMBOL, start_date=START_DATE, end_date=END_DATE)

# === Feature engineering ===
def compute_features(df):
    body_size = (df['close'] - df['open']).abs()
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    rng = df['high'] - df['low']
    rng = rng.replace(0, 1e-9)
    variation_pct = ((df['close'] - df['open']) / df['open']).fillna(0)
    df["variation_pct"] = variation_pct

    return pd.DataFrame({
        'body_size': body_size,
        'upper_wick': upper_wick,
        'lower_wick': lower_wick,
        'body_ratio': body_size / rng,
        'upper_ratio': upper_wick / rng,
        'lower_ratio': lower_wick / rng,
        'direction': np.sign(df['close'] - df['open']),
        'volume_zscore': (df['volume'] - df['volume'].rolling(1000, min_periods=1).mean()) /
                         df['volume'].rolling(1000, min_periods=1).std(ddof=0)
    }).fillna(0)

def bucket_variation(row, thresholds=(-0.01, 0.01)):
    v = row["variation_pct"]
    if v < thresholds[0]:
        return -1
    elif v > thresholds[1]:
        return 1
    return 0
def assign_candle_types(df, n_clusters=10):
    if df.empty:
        df["candle_type"] = []
        return df

    feats = compute_features(df)

    if feats.empty or len(feats) < n_clusters:
        df["candle_type"] = [0] * len(df)
        return df

    X = feats.values.astype(np.float32)
    Xw = whiten(X)

    _, labels = kmeans2(Xw, k=n_clusters, minit='++')
    df['candle_type'] = labels
    return df

df = assign_candle_types(df,10)

# === Sauvegarde ===
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Données labellisées sauvegardées dans {OUTPUT_FILE}")
