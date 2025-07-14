import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[1]))

from crypto_signals.src.predict import predict
import numpy as np

def test_calibration_fix():
    """Test the fixed calibration and threshold adjustment"""
    # Get prediction
    result = predict("BTCUSDT")
    
    print("=== PREDICTION RESULTS ===")
    print(f"Symbol: {result['symbol']}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Raw Score: (check logs)")
    print(f"Original Probability: {result['prob_up_original']}")
    print(f"Calibrated Probability: {result['prob_up']}")
    print(f"Decision Threshold: {result.get('decision_threshold', 'Not included in response')}")
    print(f"Signal: {result['signal']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Calibration Method: {result['calibration_method']}")
    
    # Verify that the signal is determined correctly based on the threshold
    # Note: We can't assert this directly since we don't know the threshold used
    # but we can print it for manual verification
    print("\n=== VERIFICATION ===")
    p_up = result['prob_up']
    print(f"Probability: {p_up}")
    print(f"Signal with threshold 0.5: {'LONG' if p_up >= 0.5 else 'SHORT'}")
    print(f"Signal with threshold 0.4: {'LONG' if p_up >= 0.4 else 'SHORT'}")
    print(f"Signal with threshold 0.3: {'LONG' if p_up >= 0.3 else 'SHORT'}")
    print(f"Actual signal: {result['signal']}")
    
    # Calculate expected value
    tp_pct = 0.004  # 0.4% take profit
    sl_pct = 0.002  # 0.2% stop loss
    ev = p_up * tp_pct - (1 - p_up) * sl_pct  # Expected value
    
    print("\n=== EXPECTED VALUE APPROACH ===")
    print(f"Expected Value: {ev:.6f}")
    print(f"EV-based Signal: {'LONG' if ev > 0 else 'SHORT'}")
    
    return result

if __name__ == "__main__":
    test_calibration_fix()