# predictor/serve_api.py
import logger
from fastapi import FastAPI, Query
from crypto_forecast_ml.data_loader import load_crypto_data
from crypto_forecast_ml.features.technical_indicators import add_technical_indicators
from crypto_forecast_ml.features.target_builder import build_targets
from crypto_forecast_ml.predictor.predict import predict_direction
from crypto_forecast_ml.data_loader import load_crypto_data_custom_range

import traceback
app = FastAPI()
import logging
import traceback
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import pandas as pd
import numpy as np
from scipy.cluster.vq import kmeans2, whiten
import os

# ✅ Initialise le logger proprement
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/predict-latest")
def predict_latest(symbol: str = Query("BTCUSDT", description="Crypto symbol")):
    try:
        logger.info(f"🔵 API called with symbol: {symbol}")
        df = load_crypto_data(symbol=symbol, days=3)
        df = add_technical_indicators(df)
        df = build_targets(df)
        df_pred = predict_direction(df)

        result = df_pred.tail(10).to_dict(orient="records")
        return {"symbol": symbol, "predictions": result}

    except Exception as e:
        logger.error("🔥 Exception occurred:")
        traceback.print_exc()  # Affiche la stack trace complète
        return {"error": str(e)}




paris = ZoneInfo("Europe/Paris")
FMT_IN = "%Y-%m-%dT%H:%M"           # 2025-07-06T09:15

@app.get("/load-data")
def load_data(
    symbol: str = Query(...),
    start_date: str = Query(..., description="YYYY-MM-DDTHH:MM 🇫🇷"),
    end_date:   str = Query(..., description="YYYY-MM-DDTHH:MM 🇫🇷")
):
    # 👉 parse + convert -> UTC
    start_utc = (datetime.strptime(start_date, FMT_IN)
                           .replace(tzinfo=paris)
                           .astimezone(timezone.utc))
    end_utc   = (datetime.strptime(end_date,   FMT_IN)
                           .replace(tzinfo=paris)
                           .astimezone(timezone.utc))

    df = load_crypto_data_custom_range(symbol=symbol,
                                       start_date=start_utc,
                                       end_date=end_utc)

    return {
        "symbol": symbol,
        "start_date": start_utc.isoformat(),
        "end_date":   end_utc.isoformat(),
        "data": df.to_dict(orient="records")
    }

def load_patterns(filename="patterns_significatifs.csv"):
    base_dir = os.path.dirname(__file__)  # dossier de ce fichier .py
    pattern_path = os.path.join(base_dir, filename)
    patterns_df = pd.read_csv(pattern_path)

    return {
        tuple(eval(row["sequence"])): {
            "bias": row["bias"],
            "bullish_ratio": row["bullish_ratio"],
            "bearish_ratio": row["bearish_ratio"],
        }
        for _, row in patterns_df.iterrows()
    }


# === Fonctions ===
def compute_features(df):
    body = (df['close'] - df['open']).abs()
    upper = df['high'] - df[['open', 'close']].max(axis=1)
    lower = df[['open', 'close']].min(axis=1) - df['low']
    rng = (df['high'] - df['low']).replace(0, 1e-9)
    return pd.DataFrame({
        'body_size': body,
        'upper_wick': upper,
        'lower_wick': lower,
        'body_ratio': body / rng,
        'upper_ratio': upper / rng,
        'lower_ratio': lower / rng,
        'direction': np.sign(df['close'] - df['open']),
        'volume_zscore': (df['volume'] - df['volume'].rolling(1000, min_periods=1).mean()) /
                         df['volume'].rolling(1000, min_periods=1).std(ddof=0)
    }).fillna(0)

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


def detect_known_patterns(df, known_patterns, max_len=3, max_results=5, min_gap_minutes=2):
    candle_ids = df['candle_type'].tolist()
    timestamps = pd.to_datetime(df['timestamp_utc']).tolist()
    matches = []

    for i in range(len(candle_ids)):
        for l in range(1, max_len + 1):
            if i + l > len(candle_ids):
                continue
            seq = tuple(candle_ids[i:i + l])
            if seq in known_patterns:
                match = {
                    "sequence": seq,
                    "start_timestamp": timestamps[i].isoformat(),
                    "end_timestamp": timestamps[i + l - 1].isoformat(),
                    "bias": known_patterns[seq]["bias"],
                    "direction": (
                        "bullish" if known_patterns[seq]["bias"] > 0.05 else
                        "bearish" if known_patterns[seq]["bias"] < -0.05 else "neutral"
                    )
                }
                matches.append(match)

    # Trier les matches par importance (bias absolu)
    matches = sorted(matches, key=lambda x: abs(x["bias"]), reverse=True)

    # Filtrer les patterns trop proches ou dupliqués
    filtered = []
    seen_sequences = set()
    last_end_time = None

    for match in matches:
        key = (match["sequence"], match["direction"])

        # Éviter les doublons exacts
        if key in seen_sequences:
            continue

        # Éviter les overlaps trop proches
        if last_end_time:
            start_time = pd.to_datetime(match["start_timestamp"])
            if (start_time - last_end_time).total_seconds() / 60 < min_gap_minutes:
                continue

        filtered.append(match)
        seen_sequences.add(key)
        last_end_time = pd.to_datetime(match["end_timestamp"])

        if len(filtered) >= max_results:
            break

    return filtered



# === Endpoint principal ===
@app.get("/load-data-patterns")
def load_data_pattern(
    symbol: str = Query(...),
    start_date: str = Query(..., description="YYYY-MM-DDTHH:MM"),
    end_date: str = Query(..., description="YYYY-MM-DDTHH:MM")
):
    # UTC parsing
    FMT = "%Y-%m-%dT%H:%M"
    start_utc = datetime.strptime(start_date, FMT).replace(tzinfo=timezone.utc)
    end_utc = datetime.strptime(end_date, FMT).replace(tzinfo=timezone.utc)

    df = load_crypto_data_custom_range(symbol=symbol, start_date=start_utc, end_date=end_utc)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    # Clustering pour obtenir candle_type
    df = assign_candle_types(df)

    # Détection des patterns dans la séquence
    patterns = detect_known_patterns(df,load_patterns())

    # À la fin de /load-patterns
    if len(patterns) > 0:
        last = patterns[-1]
        direction = last['direction']
        prob = abs(last['bias'])  # bias ∈ [-1, 1]
        short_term_forecast = {
            "direction": direction,
            "probability": min(1.0, max(0.5, 0.5 + prob)),  # simple mapping
            "bias": round(last['bias'], 3)
        }
    else:
        short_term_forecast = None

    return {
        "symbol": symbol,
        "start_date": start_utc.isoformat(),
        "end_date": end_utc.isoformat(),
        "patterns_detected": patterns,
        "short_term_forecast": short_term_forecast
    }


#uvicorn crypto_forecast_ml.predictor.serve_api:app --port 8006 --reload
