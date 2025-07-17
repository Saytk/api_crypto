import lightgbm as lgb, yaml, json
from pathlib import Path
import numpy as np
from crypto_signals.src.data_loader import load_minute
from crypto_signals.src.feature_factory import add_minute_features, FEATURE_ORDER
from crypto_signals.src.utils.custom_metrics import LogisticRegression
from crypto_signals.src.utils.logger import get_logger

log = get_logger()
CFG = yaml.safe_load(open(Path(__file__).parents[1] / "config" / "config.yaml"))

# Initialiser les dictionnaires pour stocker les modèles et métadonnées
# Ils seront chargés à la demande plutôt qu'au démarrage
MODELS = {}
METADATA = {}
CALIBRATORS = {}

def load_model(symbol: str) -> bool:
    """
    Charge un modèle et ses métadonnées à la demande.

    Args:
        symbol: Symbole de la paire (ex: BTCUSDT)

    Returns:
        bool: True si le chargement a réussi, False sinon
    """
    model_file = Path(__file__).parent / "models" / f"lgbm_hit5_{symbol}.txt"
    metadata_file = Path(__file__).parent / "models" / f"metadata_{symbol}.json"

    if not model_file.exists() or not metadata_file.exists():
        log.warning(f"Model or metadata missing for {symbol} – run train_lgbm.py first")
        return False

    try:
        # Vérifier le format du fichier
        with open(model_file, 'r') as f:
            first_line = f.readline().strip()
            if first_line != "tree":
                log.error(f"Invalid model format for {symbol}: file does not start with 'tree'")
                return False

        # Charger le modèle
        MODELS[symbol] = lgb.Booster(model_file=str(model_file))

        # Charger les métadonnées
        with open(metadata_file, 'r') as f:
            METADATA[symbol] = json.load(f)

        # Charger le calibrateur si présent
        if METADATA[symbol].get("calibration"):
            weights = np.array(METADATA[symbol]["calibration"]["weights"])
            bias = METADATA[symbol]["calibration"]["bias"]
            calibrator = LogisticRegression()
            calibrator.weights = weights
            calibrator.bias = bias
            CALIBRATORS[symbol] = calibrator

        log.info(f"Model loaded: {model_file}")
        return True
    except Exception as e:
        log.error(f"Error loading model for {symbol}: {str(e)}")
        return False

def predict(symbol: str = "BTCUSDT", use_incomplete_candle: bool = True) -> dict:
    """
    Génère une prédiction + trade suggestion à chaque minute.
    Version améliorée compatible avec le nouveau modèle GBM.

    Args:
        symbol: Symbole de la paire (ex: BTCUSDT)
        use_incomplete_candle: Si True, utilise la bougie en cours de formation

    Returns:
        dict: Prédiction et suggestion de trade
    """
    # Vérifier si le modèle est chargé et le charger si nécessaire
    if symbol not in MODELS or symbol not in METADATA:
        if not load_model(symbol):
            return {"error": f"Failed to load model for {symbol}"}

    # Récupérer les paramètres du modèle depuis les métadonnées
    metadata = METADATA[symbol]
    selected_features = metadata.get("selected_features", FEATURE_ORDER)
    window_size = metadata.get("window_size", 60)
    pred_thresh = metadata.get("pred_thresh", 0.5)

    # Charger les données
    df = load_minute(symbol, days=1)

    # Ajouter les features avec la bonne taille de fenêtre
    if use_incomplete_candle:
        # Utiliser suffisamment de données pour les features basées sur fenêtres
        feats = add_minute_features(df.tail(window_size + 30), window_size=window_size)
    else:
        feats = add_minute_features(df.tail(window_size + 31), window_size=window_size).iloc[:-1]

    # Extraire la dernière ligne pour la prédiction
    feats_last = feats.iloc[-1:]

    # Vérifier que toutes les features sélectionnées sont disponibles
    missing_features = [f for f in selected_features if f not in feats_last.columns]
    if missing_features:
        log.warning(f"Missing features: {missing_features}, using available features only")
        selected_features = [f for f in selected_features if f in feats_last.columns]

    # Faire la prédiction avec le modèle
    raw_pred = MODELS[symbol].predict(feats_last[selected_features])[0]

    # Appliquer la calibration si disponible
    if symbol in CALIBRATORS:
        calibrator = CALIBRATORS[symbol]
        p_up = calibrator.predict_proba(np.array([[raw_pred]]))[0, 1]
        log.debug(f"Applied calibration: raw={raw_pred:.4f}, calibrated={p_up:.4f}")
    else:
        p_up = raw_pred

    # Calculer la confiance (distance par rapport à 0.5)
    confidence = min(abs(p_up - 0.5) * 2, 1.0)

    # Déterminer le signal
    signal = "LONG" if p_up >= pred_thresh else "SHORT"
    last_price = df["close"].iloc[-1]

    # Calculer les niveaux de stop loss et take profit
    # Utiliser l'ATR si disponible pour des niveaux dynamiques
    if "atr_14" in feats_last.columns:
        atr = feats_last["atr_14"].iloc[0]
        sl_multiplier = 1.0  # Ajuster selon le risque souhaité
        tp_multiplier = 2.0  # Ratio risque/récompense de 1:2

        sl_pct = (atr / last_price) * sl_multiplier
        tp_pct = (atr / last_price) * tp_multiplier
    else:
        # Fallback sur des valeurs fixes
        sl_pct = 0.002  # 0.2%
        tp_pct = 0.004  # 0.4%

    # Calculer les niveaux d'entrée, SL et TP
    if signal == "LONG":
        entry = last_price
        stop_loss = round(entry * (1 - sl_pct), 4)
        take_profit = round(entry * (1 + tp_pct), 4)
    else:
        entry = last_price
        stop_loss = round(entry * (1 + sl_pct), 4)
        take_profit = round(entry * (1 - tp_pct), 4)

    # Construire la réponse
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
        "note": "Prédiction GBM améliorée avec features basées sur fenêtres"
    }
