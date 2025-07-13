"""
Unified Crypto API
-----------------
This API integrates functionality from both PA_ML and crypto_signals modules,
providing a comprehensive interface for cryptocurrency analysis, prediction,
pattern detection, and portfolio management.
"""

from fastapi import FastAPI, APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import logging

# Import authentication from crypto_signals
from crypto_signals.src.api import (
    get_current_active_user,
    User,
    Token,
    oauth2_scheme,
    login_for_access_token
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Unified Crypto API",
    description="Comprehensive API for cryptocurrency analysis, prediction, pattern detection, and portfolio management",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create routers for different modules
prediction_router = APIRouter(prefix="/prediction", tags=["Prediction"])
pattern_router = APIRouter(prefix="/pattern", tags=["Pattern Detection"])
portfolio_router = APIRouter(prefix="/portfolio", tags=["Portfolio Management"])

# ========================================================================= #
# Pydantic Models for Request/Response
# ========================================================================= #

# Portfolio Management Models
class Asset(BaseModel):
    symbol: str
    amount: float
    price: float

class Portfolio(BaseModel):
    assets: List[Asset] = []
    total_value_usd: float = 0

class PortfolioTransaction(BaseModel):
    symbol: str
    amount: float
    price: float
    transaction_type: str = Field(..., description="buy or sell")

# ========================================================================= #
# Root Endpoint
# ========================================================================= #

@app.get("/")
def read_root():
    """API root endpoint with basic information"""
    return {
        "name": "Unified Crypto API",
        "version": "1.0.0",
        "description": "Comprehensive API for cryptocurrency analysis, prediction, pattern detection, and portfolio management",
        "modules": [
            "Prediction (/prediction/*)",
            "Pattern Detection (/pattern/*)",
            "Portfolio Management (/portfolio/*)"
        ]
    }

# ========================================================================= #
# Authentication Endpoints
# ========================================================================= #

@app.post("/token", response_model=Token)
async def token(form_data = Depends(login_for_access_token)):
    """Get authentication token"""
    return form_data

# ========================================================================= #
# Prediction Endpoints (from crypto_signals)
# ========================================================================= #

@prediction_router.get("/latest")
async def predict_latest(
    symbol: str = Query("BTCUSDT", description="Ex: BTCUSDT ou ETHUSDT"),
    use_incomplete_candle: bool = Query(True, description="Utiliser la bougie en cours")
):
    """
    Returns the probability that the next minute candle will close higher,
    along with a discrete LONG / FLAT / SHORT signal.
    """
    from crypto_signals.src.predict import predict
    return predict(symbol, use_incomplete_candle)

@prediction_router.get("/historical/{symbol}")
async def get_historical_predictions(
    symbol: str = Path(..., description="Trading pair symbol"),
    days: int = Query(7, description="Number of days of historical data")
):
    """
    Returns historical predictions for a given symbol
    """
    from crypto_signals.src.api import get_historical_predictions
    return get_historical_predictions(symbol, days)

@prediction_router.get("/models")
async def get_models():
    """
    Returns the list of available prediction models
    """
    from crypto_signals.src.api import get_models
    return get_models()

@prediction_router.get("/models/{symbol}")
async def get_model_info(symbol: str):
    """
    Returns detailed information about a specific model
    """
    from crypto_signals.src.api import get_model_info
    return get_model_info(symbol)

@prediction_router.post("/backtest")
async def run_backtest(
    symbol: str = Query("BTCUSDT", description="Trading pair symbol"),
    n_folds: int = Query(5, description="Number of folds for walk-forward validation"),
    current_user: User = Depends(get_current_active_user)
):
    """
    Runs a walk-forward backtest and returns the results
    """
    from crypto_signals.src.api import BacktestRequest, run_backtest
    request = BacktestRequest(symbol=symbol, n_folds=n_folds)
    return await run_backtest(request, current_user)

@prediction_router.post("/train")
async def train_model(
    symbol: str = Query("BTCUSDT", description="Trading pair symbol"),
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    Triggers training of a new model (background task)
    """
    from crypto_signals.src.api import TrainRequest, train_model
    request = TrainRequest(symbol=symbol)
    return await train_model(request, background_tasks, current_user)

# ========================================================================= #
# Pattern Detection Endpoints (from PA_ML)
# ========================================================================= #

@pattern_router.get("/predict-latest")
async def predict_latest_pattern(symbol: str = Query("BTCUSDT", description="Crypto symbol")):
    """
    Returns the latest pattern prediction for a given symbol
    """
    from PA_ML.crypto_forecast_ml.predictor.serve_api import predict_latest
    return predict_latest(symbol)

@pattern_router.get("/load-data")
async def load_pattern_data(
    symbol: str = Query(...),
    start_date: str = Query(..., description="YYYY-MM-DDTHH:MM 🇫🇷"),
    end_date: str = Query(..., description="YYYY-MM-DDTHH:MM 🇫🇷")
):
    """
    Loads historical data for pattern analysis
    """
    from PA_ML.crypto_forecast_ml.predictor.serve_api import load_data
    return load_data(symbol, start_date, end_date)

@pattern_router.get("/load-data-patterns")
async def load_data_with_patterns(
    symbol: str = Query(...),
    start_date: str = Query(..., description="YYYY-MM-DDTHH:MM"),
    end_date: str = Query(..., description="YYYY-MM-DDTHH:MM")
):
    """
    Loads historical data and detects patterns within it
    """
    from PA_ML.crypto_forecast_ml.predictor.serve_api import load_data_pattern
    return load_data_pattern(symbol, start_date, end_date)

@pattern_router.get("/load-data-patterns-classic")
async def patterns_classic(
    symbol: str = Query(..., examples={"BTCUSDT": { "summary": "Bitcoin/USDT" }}),
    start_date: str = Query(..., description="YYYY-MM-DDTHH:MM (local)"),
    end_date: str   = Query(..., description="YYYY-MM-DDTHH:MM (local)"),
    atr_min_pct: float = Query(0.05, description="ATR filter in % (volatility guard)")
):
    """
    Loads historical data and detects patterns within it
    """
    from PA_ML.crypto_forecast_ml.predictor.serve_api import patterns_classic
    return patterns_classic(symbol, start_date, end_date,atr_min_pct)
# ========================================================================= #
# Portfolio Management Endpoints (new functionality)
# ========================================================================= #

# In-memory portfolio storage (replace with database in production)
user_portfolios = {}

@portfolio_router.get("/", response_model=Portfolio)
async def get_portfolio(current_user: User = Depends(get_current_active_user)):
    """
    Get the current user's portfolio
    """
    if current_user.username not in user_portfolios:
        user_portfolios[current_user.username] = Portfolio(assets=[], total_value_usd=0)
    
    return user_portfolios[current_user.username]

@portfolio_router.post("/transaction", response_model=Portfolio)
async def add_transaction(
    transaction: PortfolioTransaction,
    current_user: User = Depends(get_current_active_user)
):
    """
    Add a buy/sell transaction to the portfolio
    """
    if current_user.username not in user_portfolios:
        user_portfolios[current_user.username] = Portfolio(assets=[], total_value_usd=0)
    
    portfolio = user_portfolios[current_user.username]
    
    # Find if asset already exists in portfolio
    asset_exists = False
    for asset in portfolio.assets:
        if asset.symbol == transaction.symbol:
            asset_exists = True
            if transaction.transaction_type.lower() == "buy":
                # Update average price
                total_value = (asset.amount * asset.price) + (transaction.amount * transaction.price)
                asset.amount += transaction.amount
                asset.price = total_value / asset.amount if asset.amount > 0 else 0
            elif transaction.transaction_type.lower() == "sell":
                if asset.amount < transaction.amount:
                    raise HTTPException(status_code=400, detail="Not enough assets to sell")
                asset.amount -= transaction.amount
                # Remove asset if amount is 0
                if asset.amount == 0:
                    portfolio.assets = [a for a in portfolio.assets if a.symbol != transaction.symbol]
            break
    
    # If asset doesn't exist and it's a buy transaction, add it
    if not asset_exists and transaction.transaction_type.lower() == "buy":
        portfolio.assets.append(Asset(
            symbol=transaction.symbol,
            amount=transaction.amount,
            price=transaction.price
        ))
    
    # Recalculate total portfolio value
    portfolio.total_value_usd = sum(asset.amount * asset.price for asset in portfolio.assets)
    
    return portfolio

@portfolio_router.delete("/", response_model=Portfolio)
async def reset_portfolio(current_user: User = Depends(get_current_active_user)):
    """
    Reset the current user's portfolio
    """
    user_portfolios[current_user.username] = Portfolio(assets=[], total_value_usd=0)
    return user_portfolios[current_user.username]

# ========================================================================= #
# Data Endpoints (common)
# ========================================================================= #

@app.get("/data/{symbol}")
async def get_historical_data(
    symbol: str,
    days: int = Query(7, description="Number of days of historical data"),
    interval: str = Query("1m", description="Data interval (1m, 1h)")
):
    """
    Returns historical data for a given symbol
    """
    from crypto_signals.src.api import get_historical_data
    return get_historical_data(symbol, days, interval)

# Include routers in the main app
app.include_router(prediction_router)
app.include_router(pattern_router)
app.include_router(portfolio_router)

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)