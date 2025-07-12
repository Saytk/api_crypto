"""
Test script for the Crypto Signals API
This script tests the main endpoints of the API to ensure they're working correctly.
"""

import requests
import json
import time
from pprint import pprint

# API base URL - change this to match your deployment
BASE_URL = "http://localhost:8000"

def test_root():
    """Test the root endpoint"""
    print("\n=== Testing Root Endpoint ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    pprint(response.json())
    assert response.status_code == 200
    return response.json()

def test_predict():
    """Test the prediction endpoint"""
    print("\n=== Testing Prediction Endpoint ===")
    # Test with default parameters
    response = requests.get(f"{BASE_URL}/predict")
    print(f"Status Code: {response.status_code}")
    pprint(response.json())
    assert response.status_code == 200
    
    # Test with custom parameters
    params = {"symbol": "ETHUSDT", "use_incomplete_candle": "false"}
    response = requests.get(f"{BASE_URL}/predict", params=params)
    print(f"\nCustom Parameters:")
    print(f"Status Code: {response.status_code}")
    pprint(response.json())
    assert response.status_code == 200
    
    return response.json()

def test_models():
    """Test the models endpoints"""
    print("\n=== Testing Models Endpoints ===")
    # Get list of models
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status Code: {response.status_code}")
    pprint(response.json())
    assert response.status_code == 200
    models = response.json().get("available_models", [])
    
    if models:
        # Get details for the first model
        model = models[0]
        print(f"\nGetting details for model: {model}")
        response = requests.get(f"{BASE_URL}/models/{model}")
        print(f"Status Code: {response.status_code}")
        pprint(response.json())
        assert response.status_code == 200
    
    return models

def test_authentication():
    """Test the authentication flow"""
    print("\n=== Testing Authentication ===")
    # Get token
    data = {"username": "admin", "password": "secret"}
    response = requests.post(f"{BASE_URL}/token", data=data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        token_data = response.json()
        pprint(token_data)
        token = token_data.get("access_token")
        
        # Test a protected endpoint
        headers = {"Authorization": f"Bearer {token}"}
        print("\nTesting protected endpoint with token:")
        response = requests.post(
            f"{BASE_URL}/monitoring/evaluate", 
            json={"symbol": "BTCUSDT", "days": 1},
            headers=headers
        )
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            pprint(response.json())
        else:
            print(f"Error: {response.text}")
    else:
        print(f"Authentication failed: {response.text}")
    
    return response.status_code == 200

def test_data_endpoint():
    """Test the data endpoint"""
    print("\n=== Testing Data Endpoint ===")
    # Get data for BTCUSDT
    params = {"days": 1, "interval": "1m"}
    response = requests.get(f"{BASE_URL}/data/BTCUSDT", params=params)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Retrieved {data.get('count')} data points")
        if data.get('data'):
            print("Sample data point:")
            pprint(data['data'][0])
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def run_all_tests():
    """Run all tests"""
    print("Starting API tests...")
    
    tests = [
        ("Root Endpoint", test_root),
        ("Prediction Endpoint", test_predict),
        ("Models Endpoints", test_models),
        ("Data Endpoint", test_data_endpoint),
        ("Authentication", test_authentication),
    ]
    
    results = {}
    
    for name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running test: {name}")
        try:
            start_time = time.time()
            result = test_func()
            elapsed = time.time() - start_time
            results[name] = {"status": "PASS", "time": f"{elapsed:.2f}s"}
            print(f"Test {name} PASSED in {elapsed:.2f}s")
        except Exception as e:
            results[name] = {"status": "FAIL", "error": str(e)}
            print(f"Test {name} FAILED: {e}")
    
    print("\n{'='*50}")
    print("Test Summary:")
    for name, result in results.items():
        status = result["status"]
        details = f"in {result['time']}" if status == "PASS" else f"- {result['error']}"
        print(f"{name}: {status} {details}")

if __name__ == "__main__":
    run_all_tests()