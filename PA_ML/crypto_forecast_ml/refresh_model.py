# crypto_forecast_ml/refresh_model.py

from crypto_forecast_ml.data_loader import load_crypto_data
from crypto_forecast_ml.features.technical_indicators import add_technical_indicators
from crypto_forecast_ml.features.target_builder import build_targets
from crypto_forecast_ml.training.train_model import train_direction_model
from crypto_forecast_ml.predictor.predict import predict_direction

import pandas as pd
import xgboost as xgb
import os
import logging
from math import log

# ⚙️ Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_log_loss(y_true, y_pred, eps=1e-15):
    """
    Calcule le log loss sans sklearn.
    """
    y_pred = [max(min(p, 1 - eps), eps) for p in y_pred]
    loss = -sum(y * log(p) + (1 - y) * log(1 - p) for y, p in zip(y_true, y_pred)) / len(y_true)
    return loss

def evaluate_model(df: pd.DataFrame, model_path: str) -> float:
    df = df.dropna().copy()
    X = df.drop(columns=["timestamp_utc", "next_close", "return_next", "direction"])
    y_true = df["direction"].tolist()
    dmatrix = xgb.DMatrix(X)

    model = xgb.Booster()
    model.load_model(model_path)

    y_pred = model.predict(dmatrix).tolist()
    return compute_log_loss(y_true, y_pred)

def refresh_model():
    logger.info("🔁 Refresh model process started")

    df = load_crypto_data("BTCUSDT", days=3)
    df = add_technical_indicators(df)
    df = build_targets(df)

    temp_model_path = "models/xgb_direction_temp.json"
    final_model_path = "models/xgb_direction.json"
    train_direction_model(df, output_path=temp_model_path)

    try:
        old_score = evaluate_model(df, final_model_path)
    except Exception:
        old_score = float("inf")
        logger.warning("⚠️ Aucun ancien modèle valide trouvé, on utilisera le nouveau directement.")

    new_score = evaluate_model(df, temp_model_path)
    logger.info(f"📊 Old logloss: {old_score:.5f} — New logloss: {new_score:.5f}")

    if new_score < old_score:
        os.replace(temp_model_path, final_model_path)
        logger.info("✅ Nouveau modèle adopté ✅")
    else:
        os.remove(temp_model_path)
        logger.info("❌ Nouveau modèle rejeté — moins performant")

if __name__ == "__main__":
    refresh_model()
