import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[1]))

from crypto_signals.src.train_lgbm import prepare
from crypto_signals.src.feature_factory import FEATURE_ORDER
from crypto_signals.src.utils.logger import get_logger

log = get_logger()

def train_and_evaluate(X_train, y_train, X_val, y_val, params, num_rounds=800, early_stopping=50):
    """Train a model with given parameters and evaluate it"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_rounds,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)]
    )
    
    # Get predictions
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    
    # Calculate metrics
    train_precision = np.sum((y_train == 1) & (train_preds > 0.5)) / np.sum(train_preds > 0.5) if np.sum(train_preds > 0.5) > 0 else 0
    train_recall = np.sum((y_train == 1) & (train_preds > 0.5)) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 0
    
    val_precision = np.sum((y_val == 1) & (val_preds > 0.5)) / np.sum(val_preds > 0.5) if np.sum(val_preds > 0.5) > 0 else 0
    val_recall = np.sum((y_val == 1) & (val_preds > 0.5)) / np.sum(y_val == 1) if np.sum(y_val == 1) > 0 else 0
    
    # Prediction stats
    pred_stats = {
        "min": np.min(val_preds),
        "max": np.max(val_preds),
        "mean": np.mean(val_preds),
        "median": np.median(val_preds),
        "std": np.std(val_preds),
        "pct_above_0.5": np.mean(val_preds > 0.5),
        "pct_above_0.4": np.mean(val_preds > 0.4),
        "pct_above_0.3": np.mean(val_preds > 0.3),
    }
    
    return model, {
        "train_precision": train_precision,
        "train_recall": train_recall,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "pred_stats": pred_stats
    }

def test_improvements():
    # Load and prepare data
    print("Loading and preparing data...")
    df = prepare("BTCUSDT")
    
    # Split data
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    X_train = train_df[FEATURE_ORDER].values
    y_train = train_df["target"].values
    X_val = val_df[FEATURE_ORDER].values
    y_val = val_df["target"].values
    
    # Calculate class weight
    pos_rate = np.mean(y_train)
    neg_weight = 1.0
    pos_weight = 1.0 / pos_rate  # Inverse of positive rate
    
    print(f"Class balance: {pos_rate:.4f} positive, {1-pos_rate:.4f} negative")
    print(f"Calculated positive weight: {pos_weight:.2f}")
    
    # Test different configurations
    configs = [
        {
            "name": "Baseline",
            "params": {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_child_samples": 20,
                "verbosity": -1,
            }
        },
        {
            "name": "With Class Weights",
            "params": {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_child_samples": 20,
                "scale_pos_weight": pos_weight,  # Add class weight
                "verbosity": -1,
            }
        },
        {
            "name": "More Complex Model",
            "params": {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "num_leaves": 127,  # Increased complexity
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "bagging_freq": 5,
                "min_child_samples": 10,  # Reduced to allow more specific rules
                "verbosity": -1,
            }
        },
        {
            "name": "Complex Model with Class Weights",
            "params": {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "learning_rate": 0.05,
                "num_leaves": 127,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "bagging_freq": 5,
                "min_child_samples": 10,
                "scale_pos_weight": pos_weight,  # Add class weight
                "verbosity": -1,
            }
        }
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTraining {config['name']}...")
        model, metrics = train_and_evaluate(X_train, y_train, X_val, y_val, config["params"])
        results[config["name"]] = metrics
        
        # Print results
        print(f"Results for {config['name']}:")
        print(f"  Train precision@0.5: {metrics['train_precision']:.4f}")
        print(f"  Train recall@0.5: {metrics['train_recall']:.4f}")
        print(f"  Val precision@0.5: {metrics['val_precision']:.4f}")
        print(f"  Val recall@0.5: {metrics['val_recall']:.4f}")
        print(f"  Prediction stats:")
        for k, v in metrics["pred_stats"].items():
            print(f"    {k}: {v:.4f}")
    
    # Compare results
    print("\nComparison of prediction distributions:")
    for name, metrics in results.items():
        stats = metrics["pred_stats"]
        print(f"{name}: min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}, >0.5={stats['pct_above_0.5']*100:.2f}%")
    
    return results

if __name__ == "__main__":
    results = test_improvements()