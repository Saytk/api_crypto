# features/target_builder.py

import pandas as pd
import numpy as np

def build_targets(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """
    Ajoute les colonnes 'next_close', 'return_next', 'direction' comme targets.

    Args:
        df (pd.DataFrame): OHLCV + indicateurs
        horizon (int): Combien de périodes dans le futur on prédit

    Returns:
        pd.DataFrame: DataFrame avec colonnes cibles
    """
    df = df.copy()

    # Valeur future de close
    df["next_close"] = df["close"].shift(-horizon)

    # Rendement simple
    df["return_next"] = (df["next_close"] - df["close"]) / df["close"]

    # Direction (classification binaire)
    df["direction"] = (df["return_next"] > 0).astype(int)

    # Supprime les lignes avec NaN dans les colonnes cibles
    df = df.dropna(subset=["next_close", "return_next", "direction"]).reset_index(drop=True)

    # Log pour le debugging
    print(f"Colonnes après build_targets: {df.columns.tolist()}")
    print(f"Nombre de lignes après build_targets: {len(df)}")
    print(f"NaN dans next_close: {df['next_close'].isna().sum()}")
    print(f"NaN dans return_next: {df['return_next'].isna().sum()}")
    print(f"NaN dans direction: {df['direction'].isna().sum()}")

    return df
