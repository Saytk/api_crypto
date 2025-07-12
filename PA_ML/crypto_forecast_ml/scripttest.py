# crypto_forecast_ml/scripttest.py

import logging
from crypto_forecast_ml.data_loader import load_crypto_data
from crypto_forecast_ml.feature_engineering import add_features
from crypto_forecast_ml.training.train_model import train_model

# ✅ Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
)

if __name__ == "__main__":
    logging.info("🚀 Démarrage du script d'entraînement")

    # 1. Chargement des données
    df = load_crypto_data("BTCUSDT")  # tu peux changer la paire ici

    # 2. Feature engineering
    df = add_features(df)

    # 3. Entraînement du modèle
    model_path = "models/xgb_direction.json"
    train_model(df, model_path)

    logging.info(f"✅ Script terminé avec succès. Modèle sauvegardé dans {model_path}")
