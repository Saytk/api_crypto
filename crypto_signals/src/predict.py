"""
Fonctions d’inférence : génère probabilité & signal pour le dernier timestamp.
"""

import lightgbm as lgb, yaml
from pathlib import Path
from crypto_signals.src.data_loader import load_minute
from crypto_signals.src.feature_factory import add_minute_features, FEATURE_ORDER
from crypto_signals.src.utils.logger import get_logger

log = get_logger()
CFG = yaml.safe_load(open(Path(__file__).parents[1] / "config" / "config.yaml"))

# Charge tous les modèles disponibles au démarrage
MODELS = {}
for sym in CFG["assets"]:
    model_file = Path(__file__).parents[2] / "models" / f"lgbm_hit5_{sym}.txt"
    if model_file.exists():
        MODELS[sym] = lgb.Booster(model_file=model_file)
        log.info(f"Model loaded: {model_file}")
    else:
        log.warning(f"Model missing for {sym} – run train_lgbm.py first")

def predict(symbol: str = "BTCUSDT", use_incomplete_candle: bool = True) -> dict:
    """
    Génère une prédiction pour le symbole donné.

    Args:
        symbol: Le symbole à prédire (ex: "BTCUSDT")
        use_incomplete_candle: Si True, utilise la bougie en cours de formation
                              Si False, utilise uniquement la dernière bougie complète

    Returns:
        Un dictionnaire contenant la prédiction
    """
    if symbol not in MODELS:
        return {"error": f"model for {symbol} not found"}

    # Charge les données récentes (31 points pour avoir 30 points complets + la bougie en cours)
    df = load_minute(symbol, days=1)

    if use_incomplete_candle:
        # Utilise les 29 dernières bougies complètes + la bougie en cours
        # Cela permet de faire une prédiction avant la clôture de la bougie actuelle
        log.info("Utilisation de la bougie incomplète en cours pour la prédiction")
        feats = add_minute_features(df.tail(30))
    else:
        # Utilise uniquement les 30 dernières bougies complètes
        log.info("Utilisation uniquement des bougies complètes pour la prédiction")
        feats = add_minute_features(df.tail(31)).iloc[:-1]

    # Utilise la dernière ligne pour la prédiction
    feats_last = feats.iloc[-1:]
    p_up = MODELS[symbol].predict(feats_last[FEATURE_ORDER])[0]

    # Détermine le signal en fonction de la probabilité
    signal = "LONG" if p_up > 0.65 else "SHORT" if p_up < 0.35 else "FLAT"

    # Ajoute un niveau de confiance basé sur la distance aux seuils
    confidence = min(abs(p_up - 0.5) * 2, 1.0)

    return {
        "symbol": symbol,
        "timestamp": df["timestamp_utc"].iloc[-1].isoformat(),
        "prob_up": round(float(p_up), 4),
        "signal": signal,
        "confidence": round(float(confidence), 4),
        "using_incomplete_candle": use_incomplete_candle
    }
