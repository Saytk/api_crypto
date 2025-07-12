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
import gc, yaml
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
    average_precision_score
)

log = get_logger()
CFG = yaml.safe_load(open(Path(__file__).parents[1] / "config" / "config.yaml"))

# ------------------------- PARAMÈTRES « FAST » ----------------------------- #
N_TRIALS      = 50          # ←  ↓
N_ROUNDS      = 800         # ←  ↓
EARLY_STOP    = 50          # ←  ↓
PRED_THRESH   = 0.4
USE_GPU       = True        # ← active le GPU si dispo
# --------------------------------------------------------------------------- #


def prepare(symbol: str) -> pd.DataFrame:
    """Charge BigQuery puis ajoute les features minute & la target."""
    df = load_minute(symbol, days=CFG["train_days"])
    df = add_minute_features(df)
    # Target : toucher +MARGIN % dans HORIZON minutes
    MARGIN = 0.0016  # 0.16%
    HORIZON = 5      # 5 minutes

    # Calculate future max price in the next HORIZON minutes
    df["future_max"] = df["high"].rolling(HORIZON).max().shift(-HORIZON)

    # Create target: 1 if price increases by at least MARGIN% within HORIZON minutes
    df["target"] = (df["future_max"] / df["close"] > 1 + MARGIN).astype(int)

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

    # ----- Sauvegarde ------------------------------------------------------- #
    out = Path("models")
    out.mkdir(exist_ok=True)
    model_path = out / f"lgbm_hit5_{symbol}.txt"
    booster.save_model(model_path)
    log.info(f"[train] Modèle sauvegardé → {model_path}")

    # ----- Évaluations simples --------------------------------------------- #
    preds = booster.predict(X)
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
    return booster, metrics
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    train("BTCUSDT")
