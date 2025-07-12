# Simple script to test imports from custom_metrics.py
try:
    from crypto_signals.src.utils.custom_metrics import log_loss, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score, compute_class_weight
    print("Imports successful!")
except SyntaxError as e:
    print(f"Syntax error: {e}")
except Exception as e:
    print(f"Other error: {e}")