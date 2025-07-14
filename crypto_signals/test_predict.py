import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parents[1]))

from crypto_signals.src.predict import predict

def test_prediction():
    # Test prediction for BTCUSDT
    result = predict("BTCUSDT")

    print(f"Symbol: {result['symbol']}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Probability Up (Original): {result['prob_up_original']}")
    print(f"Probability Up (Calibrated): {result['prob_up']}")
    print(f"Signal: {result['signal']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Entry: {result['entry']}")
    print(f"Stop Loss: {result['stop_loss']}")
    print(f"Take Profit: {result['take_profit']}")

    # Check if the signal is correctly determined based on prob_up
    prob_up = result['prob_up']
    expected_signal = "LONG" if prob_up >= 0.5 else "SHORT"
    assert result['signal'] == expected_signal, f"Expected signal {expected_signal} but got {result['signal']}"

    # Check if stop_loss and take_profit are correctly set based on the signal
    if result['signal'] == "LONG":
        assert result['stop_loss'] < result['entry'], "Stop loss should be below entry for LONG signal"
        assert result['take_profit'] > result['entry'], "Take profit should be above entry for LONG signal"
    else:  # SHORT
        assert result['stop_loss'] > result['entry'], "Stop loss should be above entry for SHORT signal"
        assert result['take_profit'] < result['entry'], "Take profit should be below entry for SHORT signal"

    print("All assertions passed!")

if __name__ == "__main__":
    test_prediction()
