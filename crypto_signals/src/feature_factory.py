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

    # Create a copy of the dataframe to avoid the SettingWithCopyWarning
    df = df.copy()

    # Assign all values at once using a dictionary
    df = df.assign(
        minute_sin = np.sin(2 * np.pi * minute / 60),
        minute_cos = np.cos(2 * np.pi * minute / 60),
        hour_sin = np.sin(2 * np.pi * hour / 24),
        hour_cos = np.cos(2 * np.pi * hour / 24),
        dow = dow,
        day_sin = np.sin(2 * np.pi * day / 31),
        day_cos = np.cos(2 * np.pi * day / 31),
        month_sin = np.sin(2 * np.pi * month / 12),
        month_cos = np.cos(2 * np.pi * month / 12)
    )
    return df


# ========================================================================= #
# Fonction pour ajouter des features basées sur fenêtres
# ========================================================================= #
def _add_window_features(df: pd.DataFrame, window_size: int = 60) -> pd.DataFrame:
    """
    Ajoute des features basées sur une fenêtre glissante (window_size minutes).
    Ces features sont particulièrement adaptées aux modèles GBM.
    """
    # Create a copy of the dataframe to avoid the SettingWithCopyWarning
    df = df.copy()

    # Dictionary to hold all new features
    new_features = {}

    # Prix et volumes
    for col in ['close', 'high', 'low', 'volume']:
        # Statistiques de base sur la fenêtre
        new_features[f'{col}_mean_{window_size}'] = df[col].rolling(window_size).mean()
        new_features[f'{col}_std_{window_size}'] = df[col].rolling(window_size).std()
        new_features[f'{col}_min_{window_size}'] = df[col].rolling(window_size).min()
        new_features[f'{col}_max_{window_size}'] = df[col].rolling(window_size).max()

        # Quantiles pour capturer la distribution
        new_features[f'{col}_q25_{window_size}'] = df[col].rolling(window_size).quantile(0.25)
        new_features[f'{col}_q75_{window_size}'] = df[col].rolling(window_size).quantile(0.75)

    # Features de momentum (variations)
    for period in [5, 15, 30, 60]:
        if period <= window_size:
            # Variation en pourcentage
            new_features[f'pct_change_{period}'] = df['close'].pct_change(period)
            # Momentum (différence absolue)
            new_features[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
            # Accélération (dérivée seconde)
            momentum = df['close'] - df['close'].shift(period)
            new_features[f'acceleration_{period}'] = momentum - momentum.shift(period)

    # Volatilité sur différentes périodes
    for period in [5, 15, 30, 60]:
        if period <= window_size:
            new_features[f'volatility_{period}'] = df['close'].rolling(period).std() / df['close'].rolling(period).mean()

    # Ratio high/low sur différentes périodes
    for period in [5, 15, 30, 60]:
        if period <= window_size:
            new_features[f'high_low_ratio_{period}'] = df['high'].rolling(period).max() / df['low'].rolling(period).min()

    # Volume features
    new_features['volume_change_5'] = df['volume'].pct_change(5)
    new_features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window_size).mean()

    # Lags des prix de clôture (pour capturer l'autocorrélation)
    for lag in [1, 5, 15, 30]:
        new_features[f'close_lag_{lag}'] = df['close'].shift(lag)

    # Assign all new features at once
    return df.assign(**new_features)


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
    # Create a copy of the dataframe to avoid the SettingWithCopyWarning
    df = df.copy()

    # Dictionary to hold all new features
    new_features = {}

    # --------------------------------------------------------------------- #
    # Moyennes mobiles & RSI
    # --------------------------------------------------------------------- #
    new_features["sma_20"]  = df["close"].rolling(20).mean()
    new_features["sma_50"]  = df["close"].rolling(50).mean()
    new_features["sma_100"] = df["close"].rolling(100).mean()

    new_features["ema_12"]  = df["close"].ewm(span=12, adjust=False).mean()
    new_features["ema_26"]  = df["close"].ewm(span=26, adjust=False).mean()

    new_features["rsi_14"]  = ta.momentum.rsi(df["close"], window=14)

    # --------------------------------------------------------------------- #
    # Indicateurs techniques supplémentaires
    # --------------------------------------------------------------------- #
    # MACD
    macd = ta.trend.macd(df["close"])
    new_features["macd"] = macd
    new_features["macd_signal"] = ta.trend.macd_signal(df["close"])
    new_features["macd_diff"] = ta.trend.macd_diff(df["close"])

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    new_features["bb_upper"] = bb.bollinger_hband()
    new_features["bb_lower"] = bb.bollinger_lband()
    new_features["bb_width"] = bb.bollinger_wband()

    # ATR (Average True Range)
    new_features["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
    new_features["stoch_k"] = stoch.stoch()
    new_features["stoch_d"] = stoch.stoch_signal()

    # Assign the first batch of features
    df = df.assign(**new_features)

    # --------------------------------------------------------------------- #
    # Direction et force de la tendance
    # --------------------------------------------------------------------- #
    conditions_up   = (df["sma_20"] > df["sma_50"]) & (df["sma_50"] > df["sma_100"])
    conditions_down = (df["sma_20"] < df["sma_50"]) & (df["sma_50"] < df["sma_100"])

    trend_direction = np.select(
        [conditions_up, conditions_down],
        [1, -1],
        default=np.sign(df["sma_20"] - df["sma_50"])
    )

    adx_14 = ta.trend.adx(df["high"], df["low"], df["close"], window=14)

    # Add trend features
    df = df.assign(
        trend_direction=trend_direction,
        adx_14=adx_14
    )

    # --------------------------------------------------------------------- #
    # Durée de la tendance (vectorisé)
    # --------------------------------------------------------------------- #
    # Nouveau groupe chaque fois que la direction change
    trend_group = (df["trend_direction"] != df["trend_direction"].shift()).cumsum()
    # Compte cumulatif à l’intérieur de chaque groupe
    trend_duration = trend_group.groupby(trend_group).cumcount()

    df = df.assign(trend_duration=trend_duration)

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
