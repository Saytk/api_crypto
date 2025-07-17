"""
Module de surveillance et détection de drift pour les modèles de trading.

Ce module fournit des fonctionnalités pour:
1. Suivre les performances du modèle au fil du temps
2. Détecter quand un modèle commence à dériver (drift)
3. Générer des alertes lorsque des seuils de performance sont franchis
4. Recommander un réentraînement lorsque nécessaire
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional, Union

from crypto_signals.src.data_loader import load_minute
from crypto_signals.src.feature_factory import add_minute_features, FEATURE_ORDER
from crypto_signals.src.utils.logger import get_logger

log = get_logger()

# Chemin pour stocker les données de monitoring
MONITORING_PATH = Path("monitoring")
MONITORING_PATH.mkdir(exist_ok=True)


class ModelMonitor:
    """
    Classe pour surveiller les performances d'un modèle au fil du temps
    et détecter les dérives (drift).
    """

    def __init__(self, symbol: str, model_path: str, 
                 history_file: Optional[str] = None,
                 performance_threshold: float = 0.1,
                 volatility_adjustment: bool = True):
        """
        Initialise le moniteur de modèle.

        Args:
            symbol: Symbole de la paire (ex: "BTCUSDT")
            model_path: Chemin vers le fichier du modèle
            history_file: Fichier pour stocker l'historique des performances
            performance_threshold: Seuil de baisse de performance pour déclencher une alerte
            volatility_adjustment: Ajuster les seuils en fonction de la volatilité récente
        """
        self.symbol = symbol
        self.model = lgb.Booster(model_file=str(model_path))

        # Fichier d'historique par défaut
        if history_file is None:
            history_file = MONITORING_PATH / f"performance_history_{symbol}.json"
        self.history_file = Path(history_file)

        # Paramètres de détection de drift
        self.performance_threshold = performance_threshold
        self.volatility_adjustment = volatility_adjustment

        # Charger l'historique existant ou créer un nouveau
        self.performance_history = self._load_history()

    def _load_history(self) -> Dict:
        """Charge l'historique des performances ou crée un nouveau."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                log.warning(f"Erreur lors du chargement de l'historique: {e}")
                return self._create_new_history()
        else:
            return self._create_new_history()

    def _create_new_history(self) -> Dict:
        """Crée un nouvel historique vide."""
        return {
            "symbol": self.symbol,
            "model_version": datetime.now().isoformat(),
            "baseline_performance": None,
            "daily_performance": [],
            "alerts": []
        }

    def _save_history(self):
        """Sauvegarde l'historique des performances."""
        with open(self.history_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2)

    def evaluate_recent_performance(self, days: int = 1) -> Dict:
        """
        Évalue la performance du modèle sur les données récentes.

        Args:
            days: Nombre de jours de données à utiliser

        Returns:
            Dictionnaire contenant les métriques de performance
        """
        # Charger les données récentes
        df = load_minute(self.symbol, days=days)

        # Ajouter les features
        df = add_minute_features(df)

        # Calculer le target (comme dans train_lgbm.py mais sans fuite de données)
        future_high = df["high"].shift(-1).rolling(5, min_periods=5).max()
        df["target"] = (future_high >= df["close"] * (1 + 0.0016)).astype(int)
        df.dropna(inplace=True)

        if len(df) == 0:
            log.warning(f"Pas assez de données pour évaluer la performance récente")
            return {}

        # Faire des prédictions
        X = df[FEATURE_ORDER].values
        y_true = df["target"].values
        y_pred = self.model.predict(X)
        y_pred_binary = (y_pred >= 0.6).astype(int)

        # Calculer les métriques
        from crypto_signals.src.utils.custom_metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "data_points": len(df),
            "accuracy": float(accuracy_score(y_true, y_pred_binary)),
            "precision": float(precision_score(y_true, y_pred_binary, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred_binary, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred_binary, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_pred)),
            "volatility": float(df["ret_1"].std() * np.sqrt(1440))  # Volatilité annualisée
        }

        return metrics

    def update_performance_history(self, days: int = 1):
        """
        Met à jour l'historique des performances avec les données récentes.

        Args:
            days: Nombre de jours de données à utiliser
        """
        # Évaluer la performance récente
        metrics = self.evaluate_recent_performance(days)

        if not metrics:
            return

        # Ajouter à l'historique
        self.performance_history["daily_performance"].append(metrics)

        # Si c'est la première évaluation, définir comme référence
        if self.performance_history["baseline_performance"] is None:
            self.performance_history["baseline_performance"] = metrics
            log.info(f"Performance de référence établie: {metrics}")

        # Sauvegarder l'historique mis à jour
        self._save_history()

        # Vérifier s'il y a dérive
        self.check_for_drift()

    def check_for_drift(self) -> bool:
        """
        Vérifie si le modèle a dérivé en comparant les performances récentes
        avec la performance de référence.

        Returns:
            True si une dérive est détectée, False sinon
        """
        if not self.performance_history["daily_performance"] or self.performance_history["baseline_performance"] is None:
            return False

        # Obtenir les performances récentes et de référence
        recent = self.performance_history["daily_performance"][-1]
        baseline = self.performance_history["baseline_performance"]

        # Calculer le seuil ajusté en fonction de la volatilité si activé
        threshold = self.performance_threshold
        if self.volatility_adjustment:
            baseline_vol = baseline.get("volatility", 0.01)
            recent_vol = recent.get("volatility", 0.01)
            vol_ratio = recent_vol / baseline_vol if baseline_vol > 0 else 1

            # Ajuster le seuil: plus de tolérance quand la volatilité augmente
            threshold = self.performance_threshold * min(2, max(0.5, vol_ratio))
            log.info(f"Seuil ajusté pour la volatilité: {threshold:.4f} (ratio vol: {vol_ratio:.2f})")

        # Vérifier les métriques clés
        drift_detected = False
        alert_message = ""

        # Vérifier la précision
        precision_drop = baseline["precision"] - recent["precision"]
        if precision_drop > threshold:
            drift_detected = True
            alert_message += f"Baisse de précision: {precision_drop:.4f} (seuil: {threshold:.4f})\n"

        # Vérifier le F1-score
        f1_drop = baseline["f1"] - recent["f1"]
        if f1_drop > threshold:
            drift_detected = True
            alert_message += f"Baisse de F1-score: {f1_drop:.4f} (seuil: {threshold:.4f})\n"

        # Vérifier l'AUC
        auc_drop = baseline["roc_auc"] - recent["roc_auc"]
        if auc_drop > threshold:
            drift_detected = True
            alert_message += f"Baisse d'AUC: {auc_drop:.4f} (seuil: {threshold:.4f})\n"

        # Enregistrer l'alerte si dérive détectée
        if drift_detected:
            alert = {
                "timestamp": datetime.now().isoformat(),
                "message": alert_message,
                "recent_metrics": recent,
                "baseline_metrics": baseline,
                "threshold": threshold
            }
            self.performance_history["alerts"].append(alert)
            self._save_history()

            log.warning(f"ALERTE: Dérive du modèle détectée pour {self.symbol}!\n{alert_message}")
            log.warning(f"Recommandation: Réentraîner le modèle avec des données récentes.")

        return drift_detected

    def plot_performance_trend(self, metric: str = "f1", save_path: Optional[str] = None):
        """
        Génère un graphique de l'évolution des performances au fil du temps.

        Args:
            metric: Métrique à tracer ('accuracy', 'precision', 'recall', 'f1', 'roc_auc')
            save_path: Chemin où sauvegarder le graphique (optionnel)
        """
        if not self.performance_history["daily_performance"]:
            log.warning("Pas assez de données pour générer un graphique de tendance")
            return

        # Extraire les données
        history = pd.DataFrame(self.performance_history["daily_performance"])
        history["timestamp"] = pd.to_datetime(history["timestamp"])
        history = history.sort_values("timestamp")

        # Créer le graphique
        plt.figure(figsize=(12, 6))
        plt.plot(history["timestamp"], history[metric], marker='o', linestyle='-')

        # Ajouter la ligne de référence
        if self.performance_history["baseline_performance"]:
            baseline_value = self.performance_history["baseline_performance"][metric]
            plt.axhline(y=baseline_value, color='r', linestyle='--', 
                        label=f'Référence: {baseline_value:.4f}')

        # Ajouter les alertes
        for alert in self.performance_history["alerts"]:
            alert_time = pd.to_datetime(alert["timestamp"])
            plt.axvline(x=alert_time, color='orange', alpha=0.5)

        # Formater le graphique
        plt.title(f'Évolution de {metric} pour {self.symbol}')
        plt.xlabel('Date')
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        # Sauvegarder ou afficher
        if save_path:
            plt.savefig(save_path)
            log.info(f"Graphique sauvegardé: {save_path}")
        else:
            plt.show()

        plt.close()


def setup_daily_monitoring(symbols: List[str]):
    """
    Configure le monitoring quotidien pour une liste de symboles.

    Args:
        symbols: Liste des symboles à surveiller
    """
    monitors = {}

    for symbol in symbols:
        model_path = f"models/lgbm_hit5_{symbol}.txt"
        if not Path(model_path).exists():
            log.warning(f"Modèle introuvable pour {symbol}: {model_path}")
            continue

        log.info(f"Configuration du monitoring pour {symbol}")
        monitors[symbol] = ModelMonitor(symbol, model_path)

    return monitors


def run_daily_monitoring(symbols: List[str] = None):
    """
    Exécute le monitoring quotidien pour tous les symboles configurés.

    Args:
        symbols: Liste des symboles à surveiller (si None, utilise les symboles du fichier config)
    """
    import yaml
    from pathlib import Path

    # Charger la configuration si aucun symbole n'est spécifié
    if symbols is None:
        config_path = Path(__file__).parents[1] / "config" / "config.yaml"
        if config_path.exists():
            config = yaml.safe_load(open(config_path))
            symbols = config.get("assets", ["BTCUSDT", "ETHUSDT"])
        else:
            symbols = ["BTCUSDT", "ETHUSDT"]

    # Configurer les moniteurs
    monitors = setup_daily_monitoring(symbols)

    # Exécuter le monitoring pour chaque symbole
    for symbol, monitor in monitors.items():
        log.info(f"Exécution du monitoring quotidien pour {symbol}")
        monitor.update_performance_history(days=1)

        # Générer et sauvegarder le graphique de tendance
        plot_path = MONITORING_PATH / f"performance_trend_{symbol}.png"
        monitor.plot_performance_trend(save_path=str(plot_path))

    log.info(f"Monitoring quotidien terminé pour {len(monitors)} symboles")

    return monitors


if __name__ == "__main__":
    log.info("Démarrage du monitoring des modèles...")
    monitors = run_daily_monitoring()
    log.info("Monitoring terminé!")
