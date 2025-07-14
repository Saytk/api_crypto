"""
LightGBM – prédiction de « hit » (+0,16 %) en 5 min
Version FAST – 2025-07-12
- ↓ N_TRIALS  :   50   (au lieu de 200)
- ↓ N_ROUNDS  :  800   (au lieu de 2 000)
- ↓ EARLY_STOP:   50   (au lieu de 200)
- + MedianPruner Optuna
- + Option GPU (device_type="gpu") si dispo
"""

from __future__ import annotations
import gc, yaml, json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

from crypto_signals.src.utils.logger import get_logger
from crypto_signals.src.data_loader import load_minute
from crypto_signals.src.feature_factory import add_minute_features, FEATURE_ORDER
from crypto_signals.src.utils.custom_metrics import (
    roc_auc_score, log_loss, precision_score, recall_score, f1_score,
    average_precision_score, LogisticRegression
)

log = get_logger()
CFG = yaml.safe_load(open(Path(__file__).parents[1] / "config" / "config.yaml"))

# ------------------------- PARAMÈTRES « FAST » ----------------------------- #
N_TRIALS      = 50          # ←  ↓
N_ROUNDS      = 800         # ←  ↓
EARLY_STOP    = 50          # ←  ↓
PRED_THRESH   = 0.5         # ← Seuil de décision (0.5 = standard)
USE_GPU       = True        # ← active le GPU si dispo
CALIBRATE     = True        # ← active la calibration des probabilités
# --------------------------------------------------------------------------- #


def prepare(symbol: str) -> pd.DataFrame:
    """Charge BigQuery puis ajoute les features minute & la target."""
    df = load_minute(symbol, days=CFG["train_days"])
    df = add_minute_features(df)
    # Target : toucher +MARGIN % dans HORIZON minutes
    MARGIN = 0.0016  # 0.16%
    HORIZON = 5      # 5 minutes

    # Calculate future max price in the next HORIZON minutes
    # Shift by -1 first to exclude current candle, then calculate rolling max
    df["future_max"] = df["high"].shift(-1).rolling(HORIZON).max().shift(-(HORIZON-1))

    # Create target: 1 if price increases by at least MARGIN% within HORIZON minutes
    df["target"] = (df["future_max"] / df["close"] > 1 + MARGIN).astype(int)

    # Log the positive rate to understand class imbalance
    pos_rate = df["target"].mean()
    log.info(f"[prepare] Taux de classe positive: {pos_rate:.4f} ({pos_rate*100:.2f}%)")

    return df.dropna()


# --------------------------------------------------------------------------- #
def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray):
    """Fonction de coût Optuna : log_loss sur un split train/val."""
    boosting_type = trial.suggest_categorical("boosting", ["gbdt", "goss"])

    # Common parameters for all boosting types
    params = {
        "objective":        "binary",
        "metric":           "binary_logloss",
        "verbosity":        -1,
        "boosting_type":    boosting_type,
        "learning_rate":    trial.suggest_float("lr", 1e-3, 0.1, log=True),
        "num_leaves":       trial.suggest_int("leaves",   16, 128, log=True),
        "feature_fraction": trial.suggest_float("ff",     0.6, 0.95),
        "min_child_samples":trial.suggest_int("min_child",10,  100, log=True),
        "device_type":      "gpu" if USE_GPU else "cpu",
        "is_unbalance":     True,  # Gestion automatique du déséquilibre des classes
    }

    # Add bagging parameters only for gbdt (not compatible with goss)
    if boosting_type == "gbdt":
        params.update({
            "bagging_fraction": trial.suggest_float("bf", 0.6, 0.95),
            "bagging_freq":     trial.suggest_int("bfreq", 1, 10),
        })

    # Split 80 % / 20 %
    idx = int(0.8 * len(X))
    lgb_train = lgb.Dataset(X[:idx], label=y[:idx])
    lgb_val   = lgb.Dataset(X[idx:], label=y[idx:])

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=N_ROUNDS,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)]
    )

    preds = booster.predict(X[idx:])
    return log_loss(y[idx:], preds)


# --------------------------------------------------------------------------- #
def calibrate_probabilities(model, X_train, y_train, X_val, y_val):
    """
    Calibre les probabilités du modèle en utilisant une régression logistique.
    Utilise les scores bruts (log-odds) au lieu des probabilités pour éviter
    d'appliquer une sigmoid sur des valeurs déjà transformées par sigmoid.
    Retourne un modèle de calibration et les probabilités calibrées.
    """
    log.info("[calibrate] Calibration des probabilités avec régression logistique (sur scores bruts)")

    # Obtenir les scores bruts (log-odds)
    train_raw = model.predict(X_train, raw_score=True)
    val_raw = model.predict(X_val, raw_score=True)

    # Reshape pour la régression logistique (besoin d'un array 2D)
    train_raw_2d = train_raw.reshape(-1, 1)
    val_raw_2d = val_raw.reshape(-1, 1)

    # Entraîner le modèle de calibration sur les scores bruts
    calibrator = LogisticRegression()
    calibrator.fit(train_raw_2d, y_train)

    # Obtenir les probabilités calibrées
    calibrated_val_preds = calibrator.predict_proba(val_raw_2d)
    # Extraire les probabilités de la classe positive (colonne 1)
    calibrated_val_pos_preds = calibrated_val_preds[:, 1]

    # Obtenir les probabilités non calibrées pour comparaison
    val_preds = model.predict(X_val)

    # Évaluer l'amélioration
    before_log_loss = log_loss(y_val, val_preds)
    after_log_loss = log_loss(y_val, calibrated_val_pos_preds)

    log.info(f"[calibrate] Log loss avant calibration: {before_log_loss:.4f}")
    log.info(f"[calibrate] Log loss après calibration: {after_log_loss:.4f}")

    # Calculer et logger le taux de classe positive
    pos_rate = y_train.mean()
    log.info(f"[calibrate] Taux de classe positive: {pos_rate:.4f} ({pos_rate*100:.2f}%)")

    return calibrator, calibrated_val_pos_preds


def train(symbol: str = "BTCUSDT", df: pd.DataFrame | None = None):
    """Entraîne LightGBM plus rapidement grâce aux réglages ci-dessus."""
    if df is None:
        df = prepare(symbol)

    X = df[FEATURE_ORDER].values
    y = df["target"].values

    # ----- Optuna ----------------------------------------------------------- #
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study  = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(lambda t: objective(t, X, y), n_trials=N_TRIALS, show_progress_bar=False)

    best_params = study.best_params | {
        "objective":  "binary",
        "metric":     "binary_logloss",
        "verbosity":  -1,
        "device_type": "gpu" if USE_GPU else "cpu",
    }
    log.info(f"[train] Best trial {study.best_trial.number} – params {best_params}")

    # ----- Entraînement final ---------------------------------------------- #
    # Split 80 % / 20 % for validation
    idx = int(0.8 * len(X))
    lgb_train = lgb.Dataset(X[:idx], label=y[:idx])
    lgb_val = lgb.Dataset(X[idx:], label=y[idx:])

    booster = lgb.train(
        best_params,
        lgb_train,
        num_boost_round=N_ROUNDS,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                   lgb.log_evaluation(period=100)]
    )

    # ----- Calibration (optionnelle) ----------------------------------------- #
    calibrator = None
    if CALIBRATE:
        # Utiliser les mêmes indices de split que pour l'entraînement
        X_train, X_val = X[:idx], X[idx:]
        y_train, y_val = y[:idx], y[idx:]

        # Calibrer les probabilités
        calibrator, _ = calibrate_probabilities(booster, X_train, y_train, X_val, y_val)

        # Sauvegarder les paramètres du calibrateur
        # Note: Dans notre implémentation custom, les attributs sont 'weights' et 'bias'
        # mais nous utilisons les noms standards pour compatibilité
        calibration_params = {
            "weights": calibrator.weights.tolist(),
            "bias": float(calibrator.bias)
        }

        # Créer un fichier metadata pour stocker les paramètres de calibration
        metadata = {
            "model_type": "lgbm_hit5",
            "symbol": symbol,
            "pred_thresh": PRED_THRESH,
            "calibration": calibration_params,
            "training_date": pd.Timestamp.now().isoformat()
        }
    else:
        metadata = {
            "model_type": "lgbm_hit5",
            "symbol": symbol,
            "pred_thresh": PRED_THRESH,
            "calibration": None,
            "training_date": pd.Timestamp.now().isoformat()
        }

    # ----- Sauvegarde ------------------------------------------------------- #
    out = Path("models")
    out.mkdir(exist_ok=True)
    model_path = out / f"lgbm_hit5_{symbol}.txt"
    booster.save_model(model_path)

    # Sauvegarder les métadonnées (incluant les paramètres de calibration)
    metadata_path = out / f"metadata_{symbol}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    log.info(f"[train] Modèle sauvegardé → {model_path}")
    log.info(f"[train] Métadonnées sauvegardées → {metadata_path}")

    # ----- Évaluations simples --------------------------------------------- #
    raw_preds = booster.predict(X)

    # Appliquer la calibration si activée
    if CALIBRATE and calibrator:
        # predict_proba retourne un array 2D [neg_probs, pos_probs], on prend la colonne 1 (pos_probs)
        preds = calibrator.predict_proba(raw_preds.reshape(-1, 1))[:, 1]
        log.info(f"[train] Évaluation avec probabilités calibrées")
    else:
        preds = raw_preds
        log.info(f"[train] Évaluation avec probabilités brutes")

    metrics = dict(
        log_loss            = log_loss(y, preds),
        roc_auc             = roc_auc_score(y, preds),
        avg_precision_score = average_precision_score(y, preds),
        precision           = precision_score(y, preds > PRED_THRESH),
        recall              = recall_score(y, preds > PRED_THRESH),
        f1_score            = f1_score(y, preds > PRED_THRESH),
    )
    log.info(f"[train] Metrics train : {metrics}")

    gc.collect()
    return booster, calibrator, metrics
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    booster, calibrator, metrics = train("BTCUSDT")
    log.info(f"Entraînement terminé avec succès. Calibration: {'Activée' if calibrator else 'Désactivée'}")
