import lightgbm as lgb, yaml
from pathlib import Path
from crypto_signals.src.data_loader import load_minute
from crypto_signals.src.feature_factory import add_minute_features, FEATURE_ORDER
from crypto_signals.src.utils.logger import get_logger

log = get_logger()
CFG = yaml.safe_load(open(Path(__file__).parents[1] / "config" / "config.yaml"))

# Charge les modèles au démarrage
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
    Génère une prédiction + trade suggestion à chaque minute.
    """
    if symbol not in MODELS:
        return {"error": f"model for {symbol} not found"}

    df = load_minute(symbol, days=1)

    if use_incomplete_candle:
        feats = add_minute_features(df.tail(30))
    else:
        feats = add_minute_features(df.tail(31)).iloc[:-1]

    feats_last = feats.iloc[-1:]
    p_up = MODELS[symbol].predict(feats_last[FEATURE_ORDER])[0]
    confidence = min(abs(p_up - 0.5) * 2, 1.0)

    # Direction forcée
    signal = "LONG" if p_up >= 0.5 else "SHORT"
    last_price = df["close"].iloc[-1]

    # SL/TP dynamiques (à affiner plus tard)
    sl_pct = 0.002  # 0.2%
    tp_pct = 0.004  # 0.4%

    if signal == "LONG":
        entry = last_price
        stop_loss = round(entry * (1 - sl_pct), 4)
        take_profit = round(entry * (1 + tp_pct), 4)
    else:
        entry = last_price
        stop_loss = round(entry * (1 + sl_pct), 4)
        take_profit = round(entry * (1 - tp_pct), 4)

    return {
        "symbol": symbol,
        "timestamp": df["timestamp_utc"].iloc[-1].isoformat(),
        "prob_up": round(float(p_up), 4),
        "signal": signal,
        "confidence": round(float(confidence), 4),
        "using_incomplete_candle": use_incomplete_candle,
        "entry": round(float(entry), 4),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "note": "Trade suggéré automatiquement selon signal brut et variation fixe"
    }
