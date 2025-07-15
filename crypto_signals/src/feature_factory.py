"""
Génération des features minute pour le modèle « hit ».
Version améliorée – 2025-07-12
Implémente les recommandations pour GBM avec features tabulaires basées sur fenêtres.
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
# Fonction pour ajouter des features basées sur fenêtres
# ========================================================================= #
def _add_window_features(df: pd.DataFrame, window_size: int = 60) -> pd.DataFrame:
    """
    Ajoute des features basées sur une fenêtre glissante (window_size minutes).
    Ces features sont particulièrement adaptées aux modèles GBM.
    """
    # Prix et volumes
    for col in ['close', 'high', 'low', 'volume']:
        # Statistiques de base sur la fenêtre
        df[f'{col}_mean_{window_size}'] = df[col].rolling(window_size).mean()
        df[f'{col}_std_{window_size}'] = df[col].rolling(window_size).std()
        df[f'{col}_min_{window_size}'] = df[col].rolling(window_size).min()
        df[f'{col}_max_{window_size}'] = df[col].rolling(window_size).max()

        # Quantiles pour capturer la distribution
        df[f'{col}_q25_{window_size}'] = df[col].rolling(window_size).quantile(0.25)
        df[f'{col}_q75_{window_size}'] = df[col].rolling(window_size).quantile(0.75)

    # Features de momentum (variations)
    for period in [5, 15, 30, 60]:
        if period <= window_size:
            # Variation en pourcentage
            df[f'pct_change_{period}'] = df['close'].pct_change(period)
            # Momentum (différence absolue)
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
            # Accélération (dérivée seconde)
            df[f'acceleration_{period}'] = df[f'momentum_{period}'] - df[f'momentum_{period}'].shift(period)

    # Volatilité sur différentes périodes
    for period in [5, 15, 30, 60]:
        if period <= window_size:
            df[f'volatility_{period}'] = df['close'].rolling(period).std() / df['close'].rolling(period).mean()

    # Ratio high/low sur différentes périodes
    for period in [5, 15, 30, 60]:
        if period <= window_size:
            df[f'high_low_ratio_{period}'] = df['high'].rolling(period).max() / df['low'].rolling(period).min()

    # Volume features
    df['volume_change_5'] = df['volume'].pct_change(5)
    df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window_size).mean()

    # Lags des prix de clôture (pour capturer l'autocorrélation)
    for lag in [1, 5, 15, 30]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)

    return df


# ========================================================================= #
# Fonction principale : add_minute_features
# ========================================================================= #
def add_minute_features(df: pd.DataFrame, window_size: int = 60) -> pd.DataFrame:
    """
    Enrichit `df` (bougies 1 min) avec des indicateurs techniques.
    Utilise une approche de fenêtre glissante pour créer des features tabulaires
    adaptées aux modèles GBM (XGBoost/LightGBM).

    Args:
        df (pd.DataFrame): DataFrame avec données OHLCV
        window_size (int): Taille de la fenêtre glissante en minutes

    Returns:
        pd.DataFrame: DataFrame enrichi avec features techniques
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
    # Indicateurs techniques supplémentaires
    # --------------------------------------------------------------------- #
    # MACD
    macd = ta.trend.macd(df["close"])
    df["macd"] = macd
    df["macd_signal"] = ta.trend.macd_signal(df["close"])
    df["macd_diff"] = ta.trend.macd_diff(df["close"])

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()

    # ATR (Average True Range)
    df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

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
    # Features basées sur fenêtres (window-based)
    # --------------------------------------------------------------------- #
    df = _add_window_features(df, window_size)

    # --------------------------------------------------------------------- #
    # Encodage calendrier
    # --------------------------------------------------------------------- #
    df = _calendar(df)

    return df


# ========================================================================= #
# Ordre des features utilisé par le modèle
# ========================================================================= #
FEATURE_ORDER = [
    # Features de base
    "sma_20", "sma_50", "sma_100",
    "ema_12", "ema_26",
    "rsi_14",
    "adx_14", "trend_direction", "trend_duration",

    # Nouveaux indicateurs techniques
    "macd", "macd_signal", "macd_diff",
    "bb_upper", "bb_lower", "bb_width",
    "atr_14",
    "stoch_k", "stoch_d",

    # Features basées sur fenêtres (window-based)
    "close_mean_60", "close_std_60", "close_min_60", "close_max_60",
    "close_q25_60", "close_q75_60",
    "high_mean_60", "high_std_60", "high_max_60",
    "low_mean_60", "low_std_60", "low_min_60",
    "volume_mean_60", "volume_std_60",

    # Features de momentum
    "pct_change_5", "pct_change_15", "pct_change_30", "pct_change_60",
    "momentum_5", "momentum_15", "momentum_30",
    "acceleration_5", "acceleration_15",

    # Features de volatilité
    "volatility_15", "volatility_30",
    "high_low_ratio_15", "high_low_ratio_30",

    # Features de volume
    "volume_change_5", "volume_ma_ratio",

    # Lags
    "close_lag_1", "close_lag_5", "close_lag_15",

    # Features calendaires
    "minute_sin", "minute_cos",
    "hour_sin", "hour_cos",
    "dow",
    "day_sin", "day_cos",
    "month_sin", "month_cos",
]
