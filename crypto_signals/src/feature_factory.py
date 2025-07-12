"""
Génération des features minute pour le modèle « hit ».
Version corrigée – 2025-07-12
"""

from __future__ import annotations

# ========================================================================= #
# Imports
# ========================================================================= #
import pandas as pd
import numpy as np
import ta
from crypto_signals.src.utils.logger import get_logger

log = get_logger()

# ========================================================================= #
# Fonctions calendrier
# ========================================================================= #
def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp_utc"]
    minute = ts.dt.minute
    hour   = ts.dt.hour
    day    = ts.dt.day
    month  = ts.dt.month
    dow    = ts.dt.dayofweek

    df["minute_sin"] = np.sin(2 * np.pi * minute / 60)
    df["minute_cos"] = np.cos(2 * np.pi * minute / 60)
    df["hour_sin"]   = np.sin(2 * np.pi * hour   / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * hour   / 24)
    df["dow"]        = dow
    df["day_sin"]    = np.sin(2 * np.pi * day    / 31)
    df["day_cos"]    = np.cos(2 * np.pi * day    / 31)
    df["month_sin"]  = np.sin(2 * np.pi * month  / 12)
    df["month_cos"]  = np.cos(2 * np.pi * month  / 12)
    return df


# ========================================================================= #
# Fonction principale : add_minute_features
# ========================================================================= #
def add_minute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit `df` (bougies 1 min) avec des indicateurs techniques.
    Retourne le DataFrame enrichi.
    """
    # --------------------------------------------------------------------- #
    # Moyennes mobiles & RSI
    # --------------------------------------------------------------------- #
    df["sma_20"]  = df["close"].rolling(20).mean()
    df["sma_50"]  = df["close"].rolling(50).mean()
    df["sma_100"] = df["close"].rolling(100).mean()

    df["ema_12"]  = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"]  = df["close"].ewm(span=26, adjust=False).mean()

    df["rsi_14"]  = ta.momentum.rsi(df["close"], window=14)

    # --------------------------------------------------------------------- #
    # Direction et force de la tendance
    # --------------------------------------------------------------------- #
    conditions_up   = (df["sma_20"] > df["sma_50"]) & (df["sma_50"] > df["sma_100"])
    conditions_down = (df["sma_20"] < df["sma_50"]) & (df["sma_50"] < df["sma_100"])

    df["trend_direction"] = np.select(
        [conditions_up, conditions_down],
        [1, -1],
        default=np.sign(df["sma_20"] - df["sma_50"])
    )

    df["adx_14"] = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

    # --------------------------------------------------------------------- #
    # Durée de la tendance (vectorisé)
    # --------------------------------------------------------------------- #
    # Nouveau groupe chaque fois que la direction change
    trend_group = (df["trend_direction"] != df["trend_direction"].shift()).cumsum()
    # Compte cumulatif à l’intérieur de chaque groupe
    df["trend_duration"] = trend_group.groupby(trend_group).cumcount()

    # --------------------------------------------------------------------- #
    # Encodage calendrier
    # --------------------------------------------------------------------- #
    df = _calendar(df)

    return df


# ========================================================================= #
# Ordre des features utilisé par le modèle
# ========================================================================= #
FEATURE_ORDER = [
    "sma_20", "sma_50", "sma_100",
    "ema_12", "ema_26",
    "rsi_14",
    "adx_14", "trend_direction", "trend_duration",
    "minute_sin", "minute_cos",
    "hour_sin", "hour_cos",
    "dow",
    "day_sin", "day_cos",
    "month_sin", "month_cos",
]
