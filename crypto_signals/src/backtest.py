"""
Back-test walk-forward LightGBM « hit +0,16 % en 5 min »
© 2025 – Crypto Signals
"""

from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import pandas as pd

# --- pipeline d’entrainement et features --------------------------------- #
from crypto_signals.src.train_lgbm import prepare, train, FEATURE_ORDER

# ------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)8s | %(message)s"
)
log = logging.getLogger(__name__)

SIGNAL_THRESHOLD = 0.4        # même seuil que train_lgbm.PRED_THRESH
N_FOLDS          = 5          # nombre de fenêtres walk-forward
SYMBOL_DEFAULT   = "BTCUSDT"


# ------------------------------------------------------------------------- #
def _metrics(y_true: np.ndarray, y_prob: np.ndarray, thresh: float = 0.4) -> dict:
    """Calcule les métriques principales sans scikit-learn."""
    y_pred = (y_prob >= thresh).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)

    # log-loss (clip pour éviter log 0)
    p = np.clip(y_prob, 1e-12, 1 - 1e-12)
    log_loss = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)).mean()

    # ROC-AUC (implémentation simple à base de rangs)
    order = np.argsort(p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(p))
    pos = y_true == 1
    neg = ~pos
    n_pos = pos.sum()
    n_neg = neg.sum()
    auc = ((ranks[pos].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg + 1e-12))

    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, f1=f1,
        roc_auc=auc, log_loss=log_loss
    )


# ------------------------------------------------------------------------- #
def walk_forward_backtest(symbol: str = SYMBOL_DEFAULT,
                          n_folds: int = N_FOLDS) -> pd.DataFrame:
    """
    Exécute un back-test walk-forward *n_folds* segments chronologiques.
    Retourne un DataFrame récapitulatif des métriques par fold.
    """
    log.info(f"[{symbol}] préparation des données…")
    df = prepare(symbol)                       # features + target déjà nettoyés
    n_rows = len(df)
    fold_size = n_rows // (n_folds + 1)

    results = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_end  = train_end + fold_size

        train_df = df.iloc[:train_end].copy()
        test_df  = df.iloc[train_end:test_end].copy()

        log.info(f"[{symbol}] Fold {fold+1}/{n_folds} – "
                 f"train {len(train_df):,} | test {len(test_df):,}")

        # -- entraînement (sans fuite : on passe train_df à train) ---------- #
        model, _ = train(symbol=symbol, df=train_df)

        # -- prédiction sur la fenêtre test -------------------------------- #
        proba = model.predict(test_df[FEATURE_ORDER].values)
        m     = _metrics(test_df["target"].values, proba, SIGNAL_THRESHOLD)
        m["fold"] = fold + 1
        results.append(m)

    return pd.DataFrame(results)


# ------------------------------------------------------------------------- #
def main():
    symbol = SYMBOL_DEFAULT
    res_df = walk_forward_backtest(symbol, N_FOLDS)

    log.info("\n" + "-" * 60)
    log.info(f"[{symbol}] Résultats par fold :\n{res_df}")

    agg = res_df.mean(numeric_only=True)
    log.info("-" * 60)
    log.info(f"[{symbol}] Moyenne folds : "
             f"precision {agg['precision']:.3f} | "
             f"recall {agg['recall']:.3f} | "
             f"F1 {agg['f1']:.3f} | "
             f"AUC {agg['roc_auc']:.3f} | "
             f"log_loss {agg['log_loss']:.3f}")
    log.info("-" * 60)


# ------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
