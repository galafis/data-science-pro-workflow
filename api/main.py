"""FastAPI REST API for ML Model Prediction Service.

This module provides a professional FastAPI application with endpoints for:
- Health check (/health) - GET endpoint to verify API status
- Model prediction (/predict) - POST endpoint for ML predictions with explanations

The API integrates with PredictService from dsworkflows.models.predict module
to provide real-time ML predictions with probabilities and SHAP explanations.

Features:
- Automatic OpenAPI/Swagger documentation
- Input validation using Pydantic models
- Error handling and logging
- CORS support for web applications
- Detailed response models with predictions, probabilities, and explanations

Usage:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    
Then visit:
    - http://localhost:8000/docs for Swagger UI
    - http://localhost:8000/health for health check
    - POST http://localhost:8000/predict with data for predictions

Author: Gabriel Demetrios Lafis (@galafis)
Date: September 30, 2025
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Import PredictService - adjust path as needed for your environment
try:
    from dsworkflows.models.predict import PredictService
except ImportError:
    # Alternative import if running from different directory structure
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from dsworkflows.models.predict import PredictService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app configuration
app = FastAPI(
    title="ML Model Prediction API",
    description="Professional REST API for machine learning model predictions with explanations",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global prediction service instance
predict_service: Optional[PredictService] = None

# Default model path - can be overridden via environment variable
DEFAULT_MODEL_PATH = os.getenv(
    "MODEL_PATH", 
    str(Path(__file__).parent.parent / "models" / "trained_model.pkl")
)

# ===== Pydantic Models for Request/Response =====

class PredictionRequest(BaseModel):
    """Request model for prediction endpoint.
    
    Attributes:
        data: Input features as a dictionary or list of dictionaries for batch prediction
        feature_names: Optional list of feature names (required if data is array-like)
        return_proba: Whether to include class probabilities in response
        explain: Whether to include SHAP/importance explanations
        model_path: Optional path to specific model file (overrides default)
    """
    data: Union[Dict[str, Union[float, int, str]], List[Dict[str, Union[float, int, str]]], List[List[Union[float, int]]]]
    feature_names: Optional[List[str]] = None
    return_proba: bool = Field(default=True, description="Include prediction probabilities")
    explain: bool = Field(default=True, description="Include SHAP explanations")
    model_path: Optional[str] = Field(default=None, description="Path to model file")
    
    @validator('data')
    def validate_data(cls, v):
        """Ensure data is not empty."""
        if isinstance(v, dict) and not v:
            raise ValueError("Data dictionary cannot be empty")
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Data list cannot be empty")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "feature_1": 2.5,
                    "feature_2": 1.8,
                    "feature_3": 0.7,
                    "feature_4": 3.2
                },
                "return_proba": True,
                "explain": True
            }
        }

class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(description="API health status")
    message: str = Field(description="Detailed status message")
    model_loaded: bool = Field(description="Whether ML model is loaded and ready")
    model_info: Optional[Dict[str, Any]] = Field(description="Model metadata if available")

class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    success: bool = Field(description="Whether prediction was successful")
    prediction: Optional[Union[str, int, float]] = Field(description="Single prediction result")
    predictions: Optional[List[Union[str, int, float]]] = Field(description="Batch prediction results")
    probabilities: Optional[Union[Dict[str, float], List[Dict[str, float]]]] = Field(
        description="Class probabilities (single or batch)"
    )
    explanation: Optional[Dict[str, float]] = Field(description="Feature importance explanation (single)")
    explanations: Optional[List[Dict[str, float]]] = Field(description="Feature importance explanations (batch)")
    model_info: Optional[Dict[str, Any]] = Field(description="Model metadata")
    message: Optional[str] = Field(description="Additional information or warnings")

# ===== API Endpoints =====

@app.on_event("startup")
async def startup_event():
    """Initialize the prediction service on startup."""
    global predict_service
    try:
        if os.path.exists(DEFAULT_MODEL_PATH):
            predict_service = PredictService(DEFAULT_MODEL_PATH)
            logger.info(f"Model loaded successfully from {DEFAULT_MODEL_PATH}")
        else:
            logger.warning(f"Default model not found at {DEFAULT_MODEL_PATH}")
            logger.info("API will accept model_path in requests or load models on-demand")
    except Exception as e:
        logger.error(f"Failed to load default model: {e}")
        logger.info("API will load models on-demand")

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint.
    
    Returns:
        HealthResponse with API status and model information
    """
    model_loaded = predict_service is not None
    model_info = None
    
    if model_loaded:
        model_info = {
            "model_type": getattr(predict_service, 'model_type', 'Unknown'),
            "training_timestamp": getattr(predict_service, 'training_timestamp', 'Unknown'),
            "model_path": getattr(predict_service, 'model_path', 'Unknown')
        }
    
    return HealthResponse(
        status="healthy" if model_loaded else "ready",
        message="API is running and model is loaded" if model_loaded else "API is running, no default model loaded",
        model_loaded=model_loaded,
        model_info=model_info
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict endpoint for ML model inference.
    
    Args:
        request: PredictionRequest containing input data and options
        
    Returns:
        PredictionResponse with predictions, probabilities, and explanations
        
    Raises:
        HTTPException: If model loading fails or prediction errors occur
    """
    global predict_service
    
    # Determine which model to use
    model_path = request.model_path or DEFAULT_MODEL_PATH
    
    # Load model if not already loaded or if different model requested
    if (predict_service is None or 
        (request.model_path and getattr(predict_service, 'model_path', None) != model_path)):
        
        try:
            if not os.path.exists(model_path):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model file not found: {model_path}"
                )
            
            predict_service = PredictService(model_path)
            logger.info(f"Loaded model from {model_path}")
        
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load model: {str(e)}"
            )
    
    # Perform prediction
    try:
        # Check if single sample or batch
        is_batch = isinstance(request.data, list) and (
            len(request.data) > 1 or 
            (len(request.data) == 1 and isinstance(request.data[0], dict))
        )
        
        if is_batch:
            # Batch prediction
            result = predict_service.predict_batch(
                X=request.data,
                feature_names=request.feature_names,
                return_proba=request.return_proba,
                explain=request.explain
            )
            
            return PredictionResponse(
                success=True,
                predictions=result["predictions"],
                probabilities=result.get("probabilities"),
                explanations=result.get("explanations"),
                model_info={
                    "model_type": predict_service.model_type,
                    "training_timestamp": predict_service.training_timestamp
                }
            )
        
        else:
            # Single prediction
            result = predict_service.predict_one(
                x=request.data,
                feature_names=request.feature_names,
                return_proba=request.return_proba,
                explain=request.explain
            )
            
            return PredictionResponse(
                success=True,
                prediction=result["prediction"],
                probabilities=result.get("probabilities"),
                explanation=result.get("explanation"),
                model_info={
                    "model_type": predict_service.model_type,
                    "training_timestamp": predict_service.training_timestamp
                }
            )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

# ===== Exception Handlers =====

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "status_code": 500
        }
    )

# ===== Development Server =====

if __name__ == "__main__":
    """Run the development server.
    
    For production, use a proper ASGI server like:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    
    Example usage with synthetic data:
    
    1. Start the server:
       python api/main.py
    
    2. Test health endpoint:
       curl http://localhost:8000/health
    
    3. Test prediction endpoint:
       curl -X POST "http://localhost:8000/predict" \
            -H "Content-Type: application/json" \
            -d '{
                "data": {
                    "feature_1": 2.5,
                    "feature_2": 1.8,
                    "feature_3": 0.7,
                    "feature_4": 3.2,
                    "feature_5": 1.1,
                    "feature_6": 0.9
                },
                "return_proba": true,
                "explain": true
            }'
    
    4. Visit Swagger documentation:
       http://localhost:8000/docs
    """
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
