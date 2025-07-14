import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[1]))

from crypto_signals.src.predict import predict
import numpy as np

def test_with_adapted_threshold():
    """Test prediction with adapted threshold of 0.3 instead of 0.5"""
    # Get the original prediction
    result = predict("BTCUSDT")
    
    # Print original results
    print("=== ORIGINAL PREDICTION (threshold 0.5) ===")
    print(f"Symbol: {result['symbol']}")
    print(f"Probability Up (Original): {result['prob_up_original']}")
    print(f"Probability Up (Calibrated): {result['prob_up']}")
    print(f"Signal: {result['signal']}")
    print(f"Confidence: {result['confidence']}")
    
    # Apply adapted threshold of 0.3
    p_up = result['prob_up']
    adapted_signal = "LONG" if p_up >= 0.3 else "SHORT"
    adapted_confidence = min(abs(p_up - 0.3) / 0.3 * 2, 1.0)  # Rescaled confidence
    
    # Print adapted results
    print("\n=== ADAPTED PREDICTION (threshold 0.3) ===")
    print(f"Probability Up: {p_up}")
    print(f"Adapted Signal: {adapted_signal}")
    print(f"Adapted Confidence: {adapted_confidence:.4f}")
    
    # Compare with original
    print("\n=== COMPARISON ===")
    if adapted_signal != result['signal']:
        print(f"Signal changed from {result['signal']} to {adapted_signal}")
    else:
        print("Signal remained the same")
    
    print(f"Confidence changed from {result['confidence']} to {adapted_confidence:.4f}")
    
    # Calculate expected value
    tp_pct = 0.004  # 0.4% take profit
    sl_pct = 0.002  # 0.2% stop loss
    ev = p_up * tp_pct - (1 - p_up) * sl_pct  # Expected value
    ev_signal = "LONG" if ev > 0 else "SHORT"
    
    print("\n=== EXPECTED VALUE APPROACH ===")
    print(f"Expected Value: {ev:.6f}")
    print(f"EV-based Signal: {ev_signal}")
    
    return result, adapted_signal, ev_signal

if __name__ == "__main__":
    test_with_adapted_threshold()