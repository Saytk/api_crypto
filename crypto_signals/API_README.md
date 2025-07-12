# Crypto Signals API Documentation

## Overview

This document provides comprehensive documentation for the Crypto Signals API, which serves as the backend for the Crypto Signals frontend application. The API provides access to cryptocurrency price predictions, model information, historical data, backtesting results, and model monitoring.

## Getting Started

### Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Start the API server:
```bash
uvicorn crypto_signals.src.api:app --reload
```

The API will be available at `http://localhost:8000`.

### Interactive Documentation

FastAPI provides interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Authentication

The API uses OAuth2 with JWT tokens for authentication.

### Getting a Token

```
POST /token
```

**Request Body:**
```json
{
  "username": "admin",
  "password": "secret"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Using the Token

Include the token in the Authorization header for protected endpoints:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

## API Endpoints

### Root

```
GET /
```

Returns basic information about the API.

### Predictions

#### Get Latest Prediction

```
GET /predict?symbol=BTCUSDT&use_incomplete_candle=true
```

**Parameters:**
- `symbol` (string, optional): Trading pair symbol (default: "BTCUSDT")
- `use_incomplete_candle` (boolean, optional): Whether to use the incomplete current candle (default: true)

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "timestamp": "2025-07-12T17:25:54.943",
  "prob_up": 0.7234,
  "signal": "LONG",
  "confidence": 0.4468,
  "using_incomplete_candle": true
}
```

#### Custom Prediction

```
POST /predict/custom
```

**Request Body:**
```json
{
  "symbol": "ETHUSDT",
  "use_incomplete_candle": false
}
```

**Response:** Same as GET /predict

#### Historical Predictions

```
GET /predict/historical/{symbol}?days=7
```

**Parameters:**
- `symbol` (string): Trading pair symbol
- `days` (integer, optional): Number of days of historical data (default: 7)

**Response:**
```json
[
  {
    "symbol": "BTCUSDT",
    "timestamp": "2025-07-12T17:20:00.000",
    "prob_up": 0.6543,
    "signal": "LONG",
    "confidence": 0.3086
  },
  ...
]
```

### Models

#### List Available Models

```
GET /models
```

**Response:**
```json
{
  "available_models": ["BTCUSDT", "ETHUSDT"],
  "count": 2
}
```

#### Get Model Information

```
GET /models/{symbol}
```

**Parameters:**
- `symbol` (string): Trading pair symbol

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "metadata": {
    "symbol": "BTCUSDT",
    "optimal_threshold": 0.5,
    "rebalance_method": "combined",
    "training_date": "2025-07-12T15:39:15.932212",
    "performance": {
      "log_loss": 0.0535,
      "roc_auc": 0.9999,
      "precision": 0.9928,
      "recall": 0.9961,
      "f1_score": 0.9945,
      ...
    }
  },
  "feature_importance": {
    "sma_20": 123.45,
    "sma_50": 98.76,
    ...
  },
  "features": ["sma_20", "sma_50", ...]
}
```

### Backtesting

#### Run Backtest

```
POST /backtest
```

**Request Body:**
```json
{
  "symbol": "BTCUSDT",
  "n_folds": 5
}
```

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "n_folds": 5,
  "results_by_fold": [
    {
      "fold": 1,
      "precision": 0.9876,
      "recall": 0.9765,
      "f1": 0.9820,
      "roc_auc": 0.9932,
      "log_loss": 0.0432,
      ...
    },
    ...
  ],
  "average_metrics": {
    "precision": 0.9845,
    "recall": 0.9732,
    "f1": 0.9788,
    "roc_auc": 0.9912,
    "log_loss": 0.0456,
    ...
  }
}
```

### Training

#### Train Model

```
POST /train
```

**Request Body:**
```json
{
  "symbol": "BTCUSDT"
}
```

**Response:**
```json
{
  "status": "training_started",
  "symbol": "BTCUSDT",
  "message": "Training started for BTCUSDT. This may take several minutes."
}
```

### Data

#### Get Historical Data

```
GET /data/{symbol}?days=7&interval=1m
```

**Parameters:**
- `symbol` (string): Trading pair symbol
- `days` (integer, optional): Number of days of historical data (default: 7)
- `interval` (string, optional): Data interval (1m, 1h) (default: "1m")

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "interval": "1m",
  "days": 7,
  "count": 10080,
  "data": [
    {
      "timestamp_utc": "2025-07-05T00:00:00.000",
      "open": 45678.90,
      "high": 45700.00,
      "low": 45650.00,
      "close": 45690.00,
      "volume": 123.45,
      "quote_volume": 5643789.00,
      "nb_trades": 567
    },
    ...
  ]
}
```

### Monitoring

#### Evaluate Model Performance

```
POST /monitoring/evaluate
```

**Request Body:**
```json
{
  "symbol": "BTCUSDT",
  "days": 1
}
```

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "days_evaluated": 1,
  "performance": {
    "precision": 0.9876,
    "recall": 0.9765,
    "f1": 0.9820,
    "roc_auc": 0.9932,
    "log_loss": 0.0432,
    ...
  }
}
```

#### Check Model Drift

```
POST /monitoring/check-drift
```

**Request Body:**
```json
{
  "symbol": "BTCUSDT",
  "days": 1
}
```

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "drift_detected": true,
  "details": {
    "drift_detected": true,
    "drift_score": 0.75,
    "metrics_degradation": {
      "precision": -0.05,
      "recall": -0.03,
      ...
    },
    "recommendation": "Consider retraining the model"
  }
}
```

#### Run Monitoring for All Models

```
POST /monitoring/run-all
```

**Response:**
```json
{
  "status": "monitoring_started",
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "message": "Monitoring started for all available models"
}
```

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- 400: Bad Request - Invalid input
- 401: Unauthorized - Authentication required
- 403: Forbidden - Insufficient permissions
- 404: Not Found - Resource not found
- 500: Internal Server Error - Server-side error

Example error response:
```json
{
  "detail": "Model for XRPUSDT not found"
}
```

## Websocket Support

Coming soon: Real-time updates via WebSocket connections.

## Rate Limiting

In production, rate limiting is applied to prevent abuse:
- 100 requests per minute for authenticated users
- 20 requests per minute for unauthenticated users

## Support

For issues or questions, please contact the development team.