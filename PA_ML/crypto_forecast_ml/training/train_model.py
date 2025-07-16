# crypto_forecast_ml/training/train_model.py

import pandas as pd
import xgboost as xgb
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta

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
    logger.info(f"Colonnes à l'entrée de train_direction_model: {df.columns.tolist()}")
    logger.info(f"Nombre de lignes à l'entrée de train_direction_model: {len(df)}")

    df = df.dropna().copy()
    logger.info(f"Colonnes après dropna: {df.columns.tolist()}")
    logger.info(f"Nombre de lignes après dropna: {len(df)}")

    # Sélection des features
    try:
        # Vérifier si toutes les colonnes nécessaires sont présentes
        missing_columns = []
        for col in ["timestamp_utc", "next_close", "return_next", "direction"]:
            if col not in df.columns:
                missing_columns.append(col)

        if missing_columns:
            logger.error(f"Colonnes manquantes: {missing_columns}")
            raise KeyError(f"Colonnes manquantes: {missing_columns}")

        X = df.drop(columns=["timestamp_utc", "next_close", "return_next", "direction"])
        y = df["direction"]
    except Exception as e:
        logger.error(f"Erreur lors de la sélection des features: {str(e)}")
        raise

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

    # Obtenir le chemin absolu pour le logging
    if os.path.isabs(output_path):
        abs_path = output_path
    else:
        # Si le chemin est relatif, le convertir en absolu
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(current_dir))
        abs_path = os.path.join(base_dir, output_path)

    model.save_model(output_path)
    logger.info(f"✅ Modèle entraîné et sauvegardé dans : {output_path}")
    logger.info(f"✅ Chemin absolu du modèle : {abs_path}")

def train_direction_model_with_timerange(symbol: str = "BTCUSDT", hours: int = None, days: int = 7, all_data: bool = False, output_path: str = "models/xgb_direction.json"):
    """
    Charge les données pour une plage horaire spécifique, puis entraîne un modèle XGBoost.

    Args:
        symbol (str): Symbole de la paire de trading (ex: "BTCUSDT")
        hours (int, optional): Nombre d'heures de données à utiliser pour l'entraînement.
                              Si spécifié, remplace le paramètre days.
        days (int): Nombre de jours de données à utiliser si hours n'est pas spécifié
        all_data (bool): Si True, utilise toutes les données disponibles sans contrainte de temps
        output_path (str): Chemin pour sauvegarder le modèle
    """
    from data_loader import load_crypto_data, load_crypto_data_custom_range, load_crypto_data_all
    from features.feature_engineering import add_all_features
    from features.target_builder import build_targets

    logger.info(f"🚀 Démarrage de l'entraînement avec plage horaire personnalisée")
    logger.info(f"Symbol: {symbol}, Hours: {hours}, Days: {days}, All data: {all_data}")

    if all_data:
        # Utilisation de toutes les données disponibles
        logger.info(f"📅 Utilisation de TOUTES les données disponibles")
        df = load_crypto_data_all(symbol)
    elif hours is not None:
        # Calcul des dates pour la plage horaire spécifiée
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=hours)

        # Format des dates pour BigQuery
        start_date_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_date_str = end_date.strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"📅 Plage horaire: {start_date_str} à {end_date_str} (UTC)")

        # Chargement des données avec plage personnalisée
        df = load_crypto_data_custom_range(symbol, start_date_str, end_date_str)
    else:
        # Utilisation de la méthode standard avec nombre de jours
        logger.info(f"📅 Utilisation des {days} derniers jours de données")
        df = load_crypto_data(symbol, days=days)

    # Feature engineering
    logger.info(f"🔧 Application du feature engineering...")
    df = add_all_features(df)

    # Construction des cibles
    logger.info(f"🎯 Construction des variables cibles...")
    df = build_targets(df)

    # Vérification des données
    logger.info(f"📊 Nombre de lignes pour l'entraînement: {len(df)}")

    # Entraînement du modèle
    logger.info(f"⚙️ Entraînement du modèle...")
    train_direction_model(df, output_path)

    return df
