# Crypto Signals API

## Overview

This is a comprehensive API for the Crypto Signals project that provides access to cryptocurrency price predictions, model information, historical data, backtesting results, and model monitoring.

## Features

- Real-time cryptocurrency price predictions
- Historical predictions and data
- Model information and metadata
- Backtesting capabilities
- Model monitoring and drift detection
- Authentication and authorization
- Comprehensive documentation

## Installation

1. Make sure you have Python 3.8+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the API

To start the API server:

```bash
uvicorn crypto_signals.src.api:app --reload
```

The API will be available at `http://localhost:8000`.

## API Documentation

Once the server is running, you can access the interactive API documentation at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

For detailed API documentation, see the [API_README.md](API_README.md) file.

## Testing the API

A test script is provided to verify that the API is working correctly:

```bash
python crypto_signals/test_api.py
```

This script tests the main endpoints of the API and provides a summary of the results.

## Authentication

The API uses OAuth2 with JWT tokens for authentication. To get a token:

```bash
curl -X POST "http://localhost:8000/token" -d "username=admin&password=secret"
```

For protected endpoints, include the token in the Authorization header:

```bash
curl -X GET "http://localhost:8000/models" -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## Development

### Project Structure

- `api.py`: Main API implementation
- `predict.py`: Prediction functionality
- `backtest.py`: Backtesting functionality
- `train_lgbm.py`: Model training
- `data_loader.py`: Data loading and processing
- `feature_factory.py`: Feature engineering
- `model_monitoring.py`: Model monitoring and drift detection

### Adding New Endpoints

To add a new endpoint:

1. Define the endpoint in `api.py`
2. Add appropriate request/response models
3. Implement the endpoint logic
4. Update the API documentation
5. Add tests for the new endpoint

## Deployment

For production deployment:

1. Use a production ASGI server like Gunicorn:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker crypto_signals.src.api:app
```

2. Set up a reverse proxy (e.g., Nginx)
3. Configure HTTPS
4. Use environment variables for sensitive information (e.g., SECRET_KEY)
5. Implement proper rate limiting

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.