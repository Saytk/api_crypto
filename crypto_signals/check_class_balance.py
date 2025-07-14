import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[1]))

from crypto_signals.src.train_lgbm import prepare
import numpy as np

def check_class_balance():
    # Prepare data for BTCUSDT
    print("Loading and preparing data...")
    df = prepare("BTCUSDT")
    
    # Check class balance
    positive_rate = df["target"].mean()
    print(f"Class balance:")
    print(f"  - Positive examples (price increases by 0.16% in 5 min): {positive_rate:.4f} ({positive_rate*100:.2f}%)")
    print(f"  - Negative examples: {1-positive_rate:.4f} ({(1-positive_rate)*100:.2f}%)")
    
    # Check distribution of model predictions if model exists
    model_path = Path(__file__).parents[1] / "models" / "lgbm_hit5_BTCUSDT.txt"
    if model_path.exists():
        print("\nAnalyzing model predictions...")
        from crypto_signals.src.feature_factory import FEATURE_ORDER
        import lightgbm as lgb
        
        # Load model
        model = lgb.Booster(model_file=model_path)
        
        # Get predictions
        X = df[FEATURE_ORDER].values
        preds = model.predict(X)
        
        # Analyze distribution
        print(f"Prediction statistics:")
        print(f"  - Min: {np.min(preds):.4f}")
        print(f"  - Max: {np.max(preds):.4f}")
        print(f"  - Mean: {np.mean(preds):.4f}")
        print(f"  - Median: {np.median(preds):.4f}")
        print(f"  - Std Dev: {np.std(preds):.4f}")
        
        # Count predictions above thresholds
        print(f"\nPrediction distribution:")
        print(f"  - Predictions > 0.5: {np.mean(preds > 0.5):.4f} ({np.mean(preds > 0.5)*100:.2f}%)")
        print(f"  - Predictions > 0.4: {np.mean(preds > 0.4):.4f} ({np.mean(preds > 0.4)*100:.2f}%)")
        print(f"  - Predictions > 0.3: {np.mean(preds > 0.3):.4f} ({np.mean(preds > 0.3)*100:.2f}%)")
        
        # Calculate precision at different thresholds
        y = df["target"].values
        for thresh in [0.3, 0.35, 0.4, 0.45, 0.5]:
            y_pred = preds > thresh
            if np.sum(y_pred) > 0:
                precision = np.sum((y == 1) & (y_pred)) / np.sum(y_pred)
                recall = np.sum((y == 1) & (y_pred)) / np.sum(y == 1)
                print(f"  - Threshold {thresh}: Precision={precision:.4f}, Recall={recall:.4f}")
            else:
                print(f"  - Threshold {thresh}: No positive predictions")

if __name__ == "__main__":
    check_class_balance()