"""
API complète pour Crypto Signals
Fournit des endpoints pour:
- Prédictions (actuelles et historiques)
- Informations sur les modèles
- Résultats de backtesting
- Données historiques
- Monitoring des modèles
"""

from fastapi import FastAPI, Query, HTTPException, Depends, status, BackgroundTasks, Path
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime, timedelta
import json
import os
from pathlib import Path as PathLib
import pandas as pd
import yaml
import logging
import jwt
from jwt.exceptions import PyJWTError

# Import project modules
from crypto_signals.src.predict import predict, MODELS
from crypto_signals.src.backtest import walk_forward_backtest
from crypto_signals.src.train_lgbm import train
from crypto_signals.src.data_loader import load_minute
from crypto_signals.src.model_monitoring import ModelMonitor, run_daily_monitoring
from crypto_signals.src.utils.logger import get_logger

# Setup logging
log = get_logger()

# Load config
CFG = yaml.safe_load(open(PathLib(__file__).parents[1] / "config" / "config.yaml"))

# Constants
MODEL_DIR = PathLib("models")
RESULTS_DIR = PathLib(__file__).parent / "results"
SECRET_KEY = "YOUR_SECRET_KEY_HERE"  # In production, use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize FastAPI app
app = FastAPI(
    title="Crypto Signals API",
    description="API complète pour accéder aux prédictions, modèles et données de Crypto Signals",
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

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ========================================================================= #
# Pydantic Models for Request/Response
# ========================================================================= #

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

class PredictionRequest(BaseModel):
    symbol: str = Field("BTCUSDT", description="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)")
    use_incomplete_candle: bool = Field(True, description="Whether to use the incomplete current candle")

class BacktestRequest(BaseModel):
    symbol: str = Field("BTCUSDT", description="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)")
    n_folds: int = Field(5, description="Number of folds for walk-forward validation")

class TrainRequest(BaseModel):
    symbol: str = Field("BTCUSDT", description="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)")

class MonitoringRequest(BaseModel):
    symbol: str = Field("BTCUSDT", description="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)")
    days: int = Field(1, description="Number of days to evaluate")

class HistoricalDataRequest(BaseModel):
    symbol: str = Field("BTCUSDT", description="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)")
    days: int = Field(7, description="Number of days of historical data to retrieve")
    interval: str = Field("1m", description="Data interval (1m, 1h)")

# ========================================================================= #
# Authentication Functions
# ========================================================================= #

# This is a mock user database - in production, use a real database
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@example.com",
        "hashed_password": "fakehashedsecret",
        "disabled": False,
    }
}

def verify_password(plain_password, hashed_password):
    # In production, use proper password hashing (e.g., bcrypt)
    return plain_password == "secret" and hashed_password == "fakehashedsecret"

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except PyJWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# ========================================================================= #
# Helper Functions
# ========================================================================= #

def get_model_metadata(symbol: str) -> dict:
    """Get metadata for a specific model"""
    metadata_path = PathLib(__file__).parent / "models" / f"metadata_{symbol}.json"
    if not metadata_path.exists():
        return {"error": f"No metadata found for {symbol}"}

    with open(metadata_path, "r") as f:
        return json.load(f)

def get_available_models() -> List[str]:
    """Get list of available models"""
    return list(MODELS.keys())

def get_historical_predictions(symbol: str, days: int = 7) -> List[dict]:
    """Generate predictions for historical data"""
    df = load_minute(symbol, days=days)
    if df.empty:
        return []

    if symbol not in MODELS:
        return [{"error": f"Model for {symbol} not found"}]

    # Add features
    from crypto_signals.src.feature_factory import add_minute_features, FEATURE_ORDER
    feats = add_minute_features(df)

    # Generate predictions
    predictions = []
    for i in range(len(feats)):
        row = feats.iloc[i:i+1]
        p_up = MODELS[symbol].predict(row[FEATURE_ORDER])[0]
        signal = "LONG" if p_up > 0.65 else "SHORT" if p_up < 0.35 else "FLAT"
        confidence = min(abs(p_up - 0.5) * 2, 1.0)

        predictions.append({
            "symbol": symbol,
            "timestamp": df["timestamp_utc"].iloc[i].isoformat(),
            "prob_up": round(float(p_up), 4),
            "signal": signal,
            "confidence": round(float(confidence), 4)
        })

    return predictions

# ========================================================================= #
# API Endpoints
# ========================================================================= #

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/")
def read_root():
    """API root endpoint with basic information"""
    return {
        "name": "Crypto Signals API",
        "version": "1.0.0",
        "description": "API complète pour accéder aux prédictions, modèles et données de Crypto Signals",
        "endpoints": [
            "/predict", "/predict/historical", "/models", "/models/{symbol}",
            "/backtest", "/data", "/monitoring"
        ]
    }

@app.get("/predict")
def predict_latest(symbol: str = Query("BTCUSDT", description="Ex: BTCUSDT ou ETHUSDT"),
                  use_incomplete_candle: bool = Query(True, description="Utiliser la bougie en cours")):
    """
    Retourne la probabilité que la prochaine bougie minute ferme plus haut,
    ainsi qu’un signal discret LONG / FLAT / SHORT.
    """
    return predict(symbol, use_incomplete_candle)

@app.post("/predict/custom")
def predict_custom(request: PredictionRequest):
    """
    Endpoint personnalisé pour les prédictions avec options avancées
    """
    return predict(request.symbol, request.use_incomplete_candle)

@app.get("/predict/historical/{symbol}")
def get_historical_predictions_endpoint(
    symbol: str = Path(..., description="Trading pair symbol"),
    days: int = Query(7, description="Number of days of historical data")
):
    """
    Retourne les prédictions historiques pour un symbole donné
    """
    return get_historical_predictions(symbol, days)

@app.get("/models")
def get_models():
    """
    Retourne la liste des modèles disponibles
    """
    models = get_available_models()
    return {
        "available_models": models,
        "count": len(models)
    }

@app.get("/models/{symbol}")
def get_model_info(symbol: str):
    """
    Retourne les informations détaillées sur un modèle spécifique
    """
    if symbol not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model for {symbol} not found")

    metadata = get_model_metadata(symbol)

    # Get feature importance if available
    feature_importance = {}
    try:
        importance = MODELS[symbol].feature_importance()
        for i, feature in enumerate(FEATURE_ORDER):
            feature_importance[feature] = float(importance[i])
    except:
        pass

    return {
        "symbol": symbol,
        "metadata": metadata,
        "feature_importance": feature_importance,
        "features": FEATURE_ORDER
    }

@app.post("/backtest")
async def run_backtest(request: BacktestRequest, current_user: User = Depends(get_current_active_user)):
    """
    Exécute un backtest walk-forward et retourne les résultats
    """
    try:
        results_df = walk_forward_backtest(request.symbol, request.n_folds)

        # Convert DataFrame to dict for JSON response
        results = results_df.to_dict(orient="records")

        # Calculate average metrics
        avg_metrics = results_df.mean(numeric_only=True).to_dict()

        return {
            "symbol": request.symbol,
            "n_folds": request.n_folds,
            "results_by_fold": results,
            "average_metrics": avg_metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

@app.post("/train")
async def train_model(
    request: TrainRequest, 
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """
    Déclenche l'entraînement d'un nouveau modèle (tâche en arrière-plan)
    """
    if request.symbol not in CFG["assets"]:
        raise HTTPException(status_code=400, detail=f"Symbol {request.symbol} not supported")

    # Start training in background
    background_tasks.add_task(train, request.symbol)

    return {
        "status": "training_started",
        "symbol": request.symbol,
        "message": f"Training started for {request.symbol}. This may take several minutes."
    }

@app.get("/data/{symbol}")
def get_historical_data(
    symbol: str,
    days: int = Query(7, description="Number of days of historical data"),
    interval: str = Query("1m", description="Data interval (1m, 1h)")
):
    """
    Retourne les données historiques pour un symbole donné
    """
    try:
        df = load_minute(symbol, days=days)

        if interval == "1h":
            from crypto_signals.src.data_loader import minute_to_hour
            df = minute_to_hour(df)

        # Convert to dict for JSON response
        data = df.to_dict(orient="records")

        # Convert timestamps to ISO format
        for item in data:
            item["timestamp_utc"] = item["timestamp_utc"].isoformat()

        return {
            "symbol": symbol,
            "interval": interval,
            "days": days,
            "count": len(data),
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load data: {str(e)}")

@app.post("/monitoring/evaluate")
async def evaluate_model_performance(
    request: MonitoringRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Évalue les performances récentes d'un modèle
    """
    try:
        model_path = str(MODEL_DIR / f"lgbm_hit5_{request.symbol}.txt")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model for {request.symbol} not found")

        monitor = ModelMonitor(request.symbol, model_path)
        performance = monitor.evaluate_recent_performance(days=request.days)

        return {
            "symbol": request.symbol,
            "days_evaluated": request.days,
            "performance": performance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/monitoring/check-drift")
async def check_model_drift(
    request: MonitoringRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Vérifie si un modèle a subi une dérive (drift)
    """
    try:
        model_path = str(MODEL_DIR / f"lgbm_hit5_{request.symbol}.txt")
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail=f"Model for {request.symbol} not found")

        monitor = ModelMonitor(request.symbol, model_path)
        monitor.update_performance_history(days=request.days)
        drift_result = monitor.check_for_drift()

        return {
            "symbol": request.symbol,
            "drift_detected": drift_result["drift_detected"],
            "details": drift_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift check failed: {str(e)}")

@app.post("/monitoring/run-all")
async def run_monitoring_for_all(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """
    Exécute le monitoring pour tous les modèles disponibles
    """
    symbols = get_available_models()
    background_tasks.add_task(run_daily_monitoring, symbols)

    return {
        "status": "monitoring_started",
        "symbols": symbols,
        "message": "Monitoring started for all available models"
    }
