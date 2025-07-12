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
START_DATE = "2025-07-08T09:00"
END_DATE = "2025-07-11T09:00"
N_CLUSTERS = 10
OUTPUT_FILE = f"{SYMBOL}_labeled_candles.csv"

# === Chargement des données ===
df = load_crypto_data_custom_range(symbol=SYMBOL, start_date=START_DATE, end_date=END_DATE)

# === Feature engineering ===
def compute_candle_features(df):
    body_size = (df['close'] - df['open']).abs()
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']
    rng = df['high'] - df['low']
    rng = rng.replace(0, 1e-9)

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

features = compute_candle_features(df)

# === Clustering avec scipy ===
X = features.values.astype(np.float32)
X_whitened = whiten(X)  # standardise les features (important pour KMeans)

centroids, labels = kmeans2(X_whitened, k=N_CLUSTERS, minit='++')

df['candle_type'] = labels

# === Sauvegarde ===
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Données labellisées sauvegardées dans {OUTPUT_FILE}")
