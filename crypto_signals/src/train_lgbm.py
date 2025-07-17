"""
LightGBM – prédiction de « hit » (+X %) en 5-15 min
Version améliorée – 2025-07-12
- Implémente les recommandations pour GBM avec features tabulaires basées sur fenêtres
- Sélection de features basée sur l'importance
- Labellisation améliorée avec seuils dynamiques basés sur la volatilité
- Optimisation du processus d'entraînement
"""

from __future__ import annotations
import gc, yaml, json, shap
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from crypto_signals.src.utils.logger import get_logger
from crypto_signals.src.data_loader import load_minute
from crypto_signals.src.feature_factory import add_minute_features, FEATURE_ORDER
from crypto_signals.src.utils.custom_metrics import (
    roc_auc_score, log_loss, precision_score, recall_score, f1_score,
    average_precision_score, LogisticRegression
)

log = get_logger()
CFG = yaml.safe_load(open(Path(__file__).parents[1] / "config" / "config.yaml"))

# ------------------------- PARAMÈTRES AMÉLIORÉS ---------------------------- #
N_TRIALS      = 50          # Nombre d'essais Optuna
N_ROUNDS      = 1000        # Nombre maximum d'itérations
EARLY_STOP    = 50          # Arrêt anticipé si pas d'amélioration
PRED_THRESH   = 0.5         # Seuil de décision (0.5 = standard)
USE_GPU       = False       # Désactive le GPU pour éviter les erreurs de création de répertoire avec caractères invalides dans boost_compute
CALIBRATE     = True        # Active la calibration des probabilités
WINDOW_SIZE   = 60          # Taille de la fenêtre glissante (1 heure)
MAX_FEATURES  = 50          # Nombre maximum de features à conserver
HORIZON_MIN   = 5           # Horizon de prédiction minimum (minutes)
HORIZON_MAX   = 15          # Horizon de prédiction maximum (minutes)
DYNAMIC_THRESHOLD = True    # Utiliser un seuil dynamique basé sur la volatilité
# --------------------------------------------------------------------------- #


def prepare(symbol: str) -> pd.DataFrame:
    """
    Charge les données, ajoute les features et crée les labels.
    Utilise une approche améliorée avec:
    - Fenêtre glissante pour les features
    - Labellisation basée sur le mouvement de prix futur
    - Seuils dynamiques basés sur la volatilité récente
    """
    # Charger plus de données pour avoir suffisamment d'historique
    df = load_minute(symbol, days=CFG["train_days"])

    # Ajouter les features avec la fenêtre glissante
    df = add_minute_features(df, window_size=WINDOW_SIZE)

    # Calculer la volatilité récente (ATR normalisé sur 24h)
    volatility_24h = df["atr_14"].rolling(1440).mean() / df["close"]

    # Déterminer le seuil dynamique basé sur la volatilité si activé
    if DYNAMIC_THRESHOLD:
        # Utiliser un multiple de la volatilité moyenne comme seuil
        # Typiquement entre 0.5x et 1.5x la volatilité quotidienne
        volatility_multiplier = 1.0
        threshold = volatility_24h * volatility_multiplier

        # Limiter le seuil entre 0.1% et 0.5% pour éviter des valeurs extrêmes
        threshold = threshold.clip(0.001, 0.005)

        log.info(f"[prepare] Seuil dynamique basé sur la volatilité: min={threshold.min():.4%}, max={threshold.max():.4%}, mean={threshold.mean():.4%}")
    else:
        # Seuil fixe traditionnel
        threshold = 0.0016  # 0.16%
        log.info(f"[prepare] Seuil fixe: {threshold:.4%}")

    # Calculer les prix futurs pour différents horizons
    for horizon in range(HORIZON_MIN, HORIZON_MAX + 1):
        # Prix maximum futur dans les prochaines 'horizon' minutes
        df[f"future_max_{horizon}"] = df["high"].shift(-1).rolling(horizon).max().shift(-(horizon-1))

        # Prix minimum futur dans les prochaines 'horizon' minutes
        df[f"future_min_{horizon}"] = df["low"].shift(-1).rolling(horizon).min().shift(-(horizon-1))

    # Créer les labels pour chaque horizon
    for horizon in range(HORIZON_MIN, HORIZON_MAX + 1):
        if DYNAMIC_THRESHOLD:
            # Utiliser le seuil dynamique spécifique à chaque ligne
            df[f"target_up_{horizon}"] = (df[f"future_max_{horizon}"] / df["close"] > (1 + threshold)).astype(int)
            df[f"target_down_{horizon}"] = (df[f"future_min_{horizon}"] / df["close"] < (1 - threshold)).astype(int)
        else:
            # Utiliser le seuil fixe
            df[f"target_up_{horizon}"] = (df[f"future_max_{horizon}"] / df["close"] > (1 + threshold)).astype(int)
            df[f"target_down_{horizon}"] = (df[f"future_min_{horizon}"] / df["close"] < (1 - threshold)).astype(int)

    # Créer la target principale (pour l'horizon principal)
    main_horizon = HORIZON_MIN  # Utiliser l'horizon minimum comme principal

    # Target: 1 = buy (up), -1 = sell (down), 0 = hold
    conditions = [
        df[f"target_up_{main_horizon}"] == 1,
        df[f"target_down_{main_horizon}"] == 1
    ]
    choices = [1, -1]
    df["target"] = np.select(conditions, choices, default=0)

    # Convertir en classification binaire pour la compatibilité avec le modèle existant
    # 1 = signal (buy ou sell), 0 = pas de signal (hold)
    df["target_binary"] = (df["target"] != 0).astype(int)

    # Log des statistiques de classe
    pos_rate = df["target_binary"].mean()
    up_rate = (df["target"] == 1).mean()
    down_rate = (df["target"] == -1).mean()

    log.info(f"[prepare] Taux de classe positive (signal): {pos_rate:.4f} ({pos_rate*100:.2f}%)")
    log.info(f"[prepare] Taux de classe 'buy': {up_rate:.4f} ({up_rate*100:.2f}%)")
    log.info(f"[prepare] Taux de classe 'sell': {down_rate:.4f} ({down_rate*100:.2f}%)")

    # Pour la compatibilité avec le code existant, on garde "target" comme colonne principale
    # mais on utilise la version binaire
    df["target"] = df["target_binary"]

    return df.dropna()


# --------------------------------------------------------------------------- #
def select_features(X: np.ndarray, y: np.ndarray, feature_names: list) -> tuple:
    """
    Sélectionne les features les plus importantes en utilisant un modèle LightGBM préliminaire.

    Args:
        X: Matrice de features
        y: Vecteur de labels
        feature_names: Liste des noms de features

    Returns:
        tuple: (X_selected, selected_feature_names, selected_indices)
    """
    log.info(f"[select_features] Sélection des features les plus importantes (max: {MAX_FEATURES})")

    # Paramètres de base pour un modèle rapide
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "device_type": "gpu" if USE_GPU else "cpu",
        "is_unbalance": True,
    }

    # Split 80% / 20%
    idx = int(0.8 * len(X))
    lgb_train = lgb.Dataset(X[:idx], label=y[:idx])
    lgb_val = lgb.Dataset(X[idx:], label=y[idx:])

    # Entraîner un modèle rapide pour obtenir les importances
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(20, verbose=False)]
    )

    # Obtenir les importances de features
    importances = model.feature_importance(importance_type='gain')
    feature_importances = list(zip(feature_names, importances))
    feature_importances.sort(key=lambda x: x[1], reverse=True)

    # Sélectionner les top N features
    top_features = feature_importances[:MAX_FEATURES]
    selected_feature_names = [f[0] for f in top_features]

    # Créer un masque pour sélectionner les colonnes
    selected_indices = [feature_names.index(name) for name in selected_feature_names]
    X_selected = X[:, selected_indices]

    # Afficher les features sélectionnées et leur importance
    log.info(f"[select_features] Top {len(selected_feature_names)} features sélectionnées:")
    for name, importance in top_features[:10]:  # Afficher seulement les 10 premières
        log.info(f"  - {name}: {importance}")

    # Sauvegarder un graphique des importances de features
    plt.figure(figsize=(10, 6))
    plt.barh([f[0] for f in top_features[:20]], [f[1] for f in top_features[:20]])
    plt.xlabel('Importance (gain)')
    plt.title('Top 20 Features par Importance')
    plt.tight_layout()

    # Créer le dossier models s'il n'existe pas
    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)

    # Sauvegarder le graphique
    plt.savefig(models_dir / f"feature_importance_{selected_feature_names[0]}.png")
    plt.close()

    return X_selected, selected_feature_names, selected_indices


def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray, feature_names: list = None):
    """
    Fonction de coût Optuna : log_loss sur un split train/val.
    Version améliorée avec plus d'hyperparamètres et meilleure gestion du déséquilibre.
    """
    boosting_type = trial.suggest_categorical("boosting", ["gbdt", "goss"])

    # Paramètres communs pour tous les types de boosting
    params = {
        "objective":        "binary",
        "metric":           "binary_logloss",
        "verbosity":        -1,
        "boosting_type":    boosting_type,
        "learning_rate":    trial.suggest_float("lr", 1e-3, 0.1, log=True),
        "num_leaves":       trial.suggest_int("leaves", 16, 128, log=True),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "feature_fraction": trial.suggest_float("ff", 0.6, 0.95),
        "min_child_samples":trial.suggest_int("min_child", 10, 100, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "device_type":      "gpu" if USE_GPU else "cpu",
    }

    # Gestion du déséquilibre des classes
    class_weight_method = trial.suggest_categorical("class_weight", ["balanced", "focal", "none"])
    if class_weight_method == "balanced":
        # Calculer le poids de la classe minoritaire
        pos_weight = (1 - y[:int(0.8 * len(y))].mean()) / y[:int(0.8 * len(y))].mean()
        params["scale_pos_weight"] = pos_weight
    elif class_weight_method == "focal":
        # Focal loss pour mieux gérer le déséquilibre
        params["boost_from_average"] = False
        params["alpha"] = trial.suggest_float("focal_alpha", 0.1, 0.9)
        params["tweedie_variance_power"] = 1.5  # Approximation de focal loss

    # Ajouter les paramètres de bagging uniquement pour gbdt (incompatible avec goss)
    if boosting_type == "gbdt":
        params.update({
            "bagging_fraction": trial.suggest_float("bf", 0.6, 0.95),
            "bagging_freq":     trial.suggest_int("bfreq", 1, 10),
        })

    # Split 80 % / 20 %
    idx = int(0.8 * len(X))
    lgb_train = lgb.Dataset(X[:idx], label=y[:idx], feature_name=feature_names)
    lgb_val   = lgb.Dataset(X[idx:], label=y[idx:], feature_name=feature_names)

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
def calibrate_probabilities(model, X_train, y_train, X_val, y_val, feature_names=None):
    """
    Calibre les probabilités du modèle en utilisant une régression logistique.
    Utilise les scores bruts (log-odds) au lieu des probabilités pour éviter
    d'appliquer une sigmoid sur des valeurs déjà transformées par sigmoid.

    Version améliorée avec:
    - Évaluation plus complète avant/après calibration
    - Visualisation de la calibration

    Args:
        model: Modèle LightGBM entraîné
        X_train: Features d'entraînement
        y_train: Labels d'entraînement
        X_val: Features de validation
        y_val: Labels de validation
        feature_names: Noms des features (optionnel)

    Returns:
        tuple: (calibrator, calibrated_val_preds)
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

    # Évaluer l'amélioration sur plusieurs métriques
    metrics_before = {
        "log_loss": log_loss(y_val, val_preds),
        "roc_auc": roc_auc_score(y_val, val_preds),
        "avg_precision": average_precision_score(y_val, val_preds),
        "precision": precision_score(y_val, val_preds > PRED_THRESH),
        "recall": recall_score(y_val, val_preds > PRED_THRESH),
        "f1": f1_score(y_val, val_preds > PRED_THRESH)
    }

    metrics_after = {
        "log_loss": log_loss(y_val, calibrated_val_pos_preds),
        "roc_auc": roc_auc_score(y_val, calibrated_val_pos_preds),
        "avg_precision": average_precision_score(y_val, calibrated_val_pos_preds),
        "precision": precision_score(y_val, calibrated_val_pos_preds > PRED_THRESH),
        "recall": recall_score(y_val, calibrated_val_pos_preds > PRED_THRESH),
        "f1": f1_score(y_val, calibrated_val_pos_preds > PRED_THRESH)
    }

    # Afficher les métriques avant/après calibration
    log.info("[calibrate] Métriques avant calibration:")
    for metric, value in metrics_before.items():
        log.info(f"  - {metric}: {value:.4f}")

    log.info("[calibrate] Métriques après calibration:")
    for metric, value in metrics_after.items():
        log.info(f"  - {metric}: {value:.4f}")

    # Calculer et logger le taux de classe positive
    pos_rate = y_train.mean()
    log.info(f"[calibrate] Taux de classe positive: {pos_rate:.4f} ({pos_rate*100:.2f}%)")

    # Créer une visualisation de la calibration (optionnel)
    try:
        from sklearn.calibration import calibration_curve

        # Calculer les courbes de calibration
        prob_true_before, prob_pred_before = calibration_curve(y_val, val_preds, n_bins=10)
        prob_true_after, prob_pred_after = calibration_curve(y_val, calibrated_val_pos_preds, n_bins=10)

        # Créer le graphique
        plt.figure(figsize=(10, 8))
        plt.plot(prob_pred_before, prob_true_before, marker='o', linewidth=1, label='Avant calibration')
        plt.plot(prob_pred_after, prob_true_after, marker='o', linewidth=1, label='Après calibration')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Parfaite calibration')
        plt.xlabel('Probabilité prédite')
        plt.ylabel('Fraction de positifs')
        plt.title('Courbe de calibration')
        plt.legend()
        plt.grid(True)

        # Sauvegarder le graphique
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(exist_ok=True)
        plt.savefig(models_dir / "calibration_curve.png")
        plt.close()
    except Exception as e:
        log.warning(f"[calibrate] Impossible de créer la courbe de calibration: {e}")

    return calibrator, calibrated_val_pos_preds


def train(symbol: str = "BTCUSDT", df: pd.DataFrame | None = None):
    """
    Entraîne un modèle LightGBM amélioré avec:
    - Sélection de features basée sur l'importance
    - Optimisation des hyperparamètres via Optuna
    - Calibration des probabilités
    - Évaluation complète des performances

    Args:
        symbol: Symbole de la paire de trading (ex: BTCUSDT)
        df: DataFrame préparé (optionnel, sinon chargé via prepare())

    Returns:
        tuple: (booster, calibrator, metrics, selected_features)
    """
    if df is None:
        df = prepare(symbol)

    # Extraire toutes les features disponibles
    X_full = df[FEATURE_ORDER].values
    y = df["target"].values

    # ----- Sélection de features -------------------------------------------- #
    log.info(f"[train] Démarrage de la sélection de features pour {symbol}")
    X, selected_features, selected_indices = select_features(X_full, y, FEATURE_ORDER)
    log.info(f"[train] {len(selected_features)}/{len(FEATURE_ORDER)} features sélectionnées")

    # ----- Optuna ----------------------------------------------------------- #
    log.info(f"[train] Démarrage de l'optimisation des hyperparamètres avec Optuna ({N_TRIALS} essais)")
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(
        lambda t: objective(t, X, y, selected_features), 
        n_trials=N_TRIALS, 
        show_progress_bar=False
    )

    best_params = study.best_params | {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "device_type": "gpu" if USE_GPU else "cpu",
    }
    log.info(f"[train] Meilleur essai {study.best_trial.number} – params {best_params}")

    # ----- Entraînement final ---------------------------------------------- #
    log.info(f"[train] Entraînement du modèle final avec les meilleurs paramètres")
    # Split 80 % / 20 % for validation
    idx = int(0.8 * len(X))
    lgb_train = lgb.Dataset(X[:idx], label=y[:idx], feature_name=selected_features)
    lgb_val = lgb.Dataset(X[idx:], label=y[idx:], feature_name=selected_features)

    booster = lgb.train(
        best_params,
        lgb_train,
        num_boost_round=N_ROUNDS,
        valid_sets=[lgb_val],
        callbacks=[
            lgb.early_stopping(EARLY_STOP, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )

    # ----- SHAP Values (optionnel) ------------------------------------------ #
    try:
        log.info(f"[train] Calcul des valeurs SHAP pour l'interprétabilité")
        # Créer un explainer SHAP
        explainer = shap.TreeExplainer(booster)

        # Calculer les valeurs SHAP sur un échantillon des données de validation
        sample_size = min(1000, X[idx:].shape[0])
        sample_indices = np.random.choice(X[idx:].shape[0], sample_size, replace=False)
        X_sample = X[idx:][sample_indices]

        # Calculer les valeurs SHAP
        shap_values = explainer.shap_values(X_sample)

        # Créer un résumé des valeurs SHAP
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X_sample, 
            feature_names=selected_features,
            show=False
        )

        # Sauvegarder le graphique
        models_dir = Path(__file__).parent / "models"
        models_dir.mkdir(exist_ok=True)
        plt.savefig(models_dir / f"shap_summary_{symbol}.png")
        plt.close()

        log.info(f"[train] Graphique SHAP sauvegardé")
    except Exception as e:
        log.warning(f"[train] Impossible de calculer les valeurs SHAP: {e}")

    # ----- Calibration (optionnelle) ----------------------------------------- #
    calibrator = None
    if CALIBRATE:
        log.info(f"[train] Calibration des probabilités")
        # Utiliser les mêmes indices de split que pour l'entraînement
        X_train, X_val = X[:idx], X[idx:]
        y_train, y_val = y[:idx], y[idx:]

        # Calibrer les probabilités
        calibrator, _ = calibrate_probabilities(
            booster, X_train, y_train, X_val, y_val, selected_features
        )

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
            "selected_features": selected_features,
            "window_size": WINDOW_SIZE,
            "horizon_min": HORIZON_MIN,
            "horizon_max": HORIZON_MAX,
            "dynamic_threshold": DYNAMIC_THRESHOLD,
            "training_date": pd.Timestamp.now().isoformat()
        }
    else:
        metadata = {
            "model_type": "lgbm_hit5",
            "symbol": symbol,
            "pred_thresh": PRED_THRESH,
            "calibration": None,
            "selected_features": selected_features,
            "window_size": WINDOW_SIZE,
            "horizon_min": HORIZON_MIN,
            "horizon_max": HORIZON_MAX,
            "dynamic_threshold": DYNAMIC_THRESHOLD,
            "training_date": pd.Timestamp.now().isoformat()
        }

    # ----- Sauvegarde ------------------------------------------------------- #
    log.info(f"[train] Sauvegarde du modèle et des métadonnées")
    out = Path(__file__).parent / "models"
    out.mkdir(exist_ok=True)
    model_path = out / f"lgbm_hit5_{symbol}.txt"

    # Sauvegarder le modèle avec gestion d'erreur
    try:
        booster.save_model(str(model_path))

        # Vérifier que le fichier a été correctement sauvegardé
        with open(model_path, 'r') as f:
            first_line = f.readline().strip()
            if first_line != "tree":
                log.error(f"[train] Le modèle sauvegardé n'a pas le bon format (ne commence pas par 'tree')")
                # Supprimer le fichier corrompu
                model_path.unlink(missing_ok=True)
                raise ValueError("Invalid model format")
    except Exception as e:
        log.error(f"[train] Erreur lors de la sauvegarde du modèle: {e}")
        raise

    # Sauvegarder les métadonnées (incluant les paramètres de calibration et features)
    metadata_path = out / f"metadata_{symbol}.json"
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        log.error(f"[train] Erreur lors de la sauvegarde des métadonnées: {e}")
        raise

    log.info(f"[train] Modèle sauvegardé → {model_path}")
    log.info(f"[train] Métadonnées sauvegardées → {metadata_path}")

    # ----- Évaluations complètes -------------------------------------------- #
    log.info(f"[train] Évaluation des performances du modèle")
    raw_preds = booster.predict(X)

    # Appliquer la calibration si activée
    if CALIBRATE and calibrator:
        # predict_proba retourne un array 2D [neg_probs, pos_probs], on prend la colonne 1 (pos_probs)
        preds = calibrator.predict_proba(raw_preds.reshape(-1, 1))[:, 1]
        log.info(f"[train] Évaluation avec probabilités calibrées")
    else:
        preds = raw_preds
        log.info(f"[train] Évaluation avec probabilités brutes")

    # Calculer les métriques sur l'ensemble du jeu de données
    metrics = dict(
        log_loss            = log_loss(y, preds),
        roc_auc             = roc_auc_score(y, preds),
        avg_precision_score = average_precision_score(y, preds),
        precision           = precision_score(y, preds > PRED_THRESH),
        recall              = recall_score(y, preds > PRED_THRESH),
        f1_score            = f1_score(y, preds > PRED_THRESH),
    )
    log.info(f"[train] Métriques globales: {metrics}")

    # Calculer les métriques sur les données de validation uniquement
    val_metrics = dict(
        log_loss            = log_loss(y[idx:], preds[idx:]),
        roc_auc             = roc_auc_score(y[idx:], preds[idx:]),
        avg_precision_score = average_precision_score(y[idx:], preds[idx:]),
        precision           = precision_score(y[idx:], preds[idx:] > PRED_THRESH),
        recall              = recall_score(y[idx:], preds[idx:] > PRED_THRESH),
        f1_score            = f1_score(y[idx:], preds[idx:] > PRED_THRESH),
    )
    log.info(f"[train] Métriques validation: {val_metrics}")

    # Libérer la mémoire
    gc.collect()

    return booster, calibrator, metrics, selected_features
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    booster, calibrator, metrics, selected_features = train("BTCUSDT")
    log.info(f"Entraînement terminé avec succès. Calibration: {'Activée' if calibrator else 'Désactivée'}")
    log.info(f"Features sélectionnées: {len(selected_features)}/{len(FEATURE_ORDER)}")
