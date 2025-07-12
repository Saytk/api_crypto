# crypto_forecast_ml/training/train_model.py

import pandas as pd
import xgboost as xgb
import os
from pathlib import Path
import logging

# ⚙️ Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_direction_model(df: pd.DataFrame, output_path: str = "models/xgb_direction.json"):
    """
    Entraîne un modèle XGBoost pour prédire la direction (hausse/baisse) sans sklearn.

    Args:
        df (pd.DataFrame): Données avec indicateurs et colonnes cibles
        output_path (str): Chemin pour sauvegarder le modèle
    """
    df = df.dropna().copy()

    # Sélection des features
    X = df.drop(columns=["timestamp_utc", "next_close", "return_next", "direction"])
    y = df["direction"]

    # Encodage dans un DMatrix (XGBoost natif)
    dtrain = xgb.DMatrix(X, label=y)

    # Paramètres XGBoost
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 1
    }

    # Entraînement
    model = xgb.train(params, dtrain, num_boost_round=100)

    # Sauvegarde du modèle
    os.makedirs(Path(output_path).parent, exist_ok=True)
    model.save_model(output_path)
    logger.info(f"✅ Modèle entraîné et sauvegardé dans : {output_path}")
