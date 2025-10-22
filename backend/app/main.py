# backend/app/main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import Dict, Any, Optional
from .model_manager import load_model_from_wandb, get_model_info, validate_model_performance

app = FastAPI(title="Iris inference service", version="2.0.0")

class PredictRequest(BaseModel):
    # expects list of 4 floats (sepal/petal measurements)
    features: list

class ModelInfo(BaseModel):
    artifact_name: str
    artifact_version: Optional[str] = None
    artifact_type: Optional[str] = None
    artifact_description: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@app.on_event("startup")
def startup_load_model():
    global model, model_metadata
    wandb_project = os.getenv("WANDB_PROJECT", "mlops-capstone")
    wandb_entity = os.getenv("WANDB_ENTITY", None)
    try:
        model, model_metadata = load_model_from_wandb(wandb_entity, wandb_project)
        app.state.model_ready = True
        app.state.model_metadata = model_metadata
        print(f"✅ Model loaded successfully with metadata: {model_metadata.get('test_accuracy', 'N/A')}")
    except Exception as e:
        # If model not available, mark not ready but keep service up.
        model = None
        model_metadata = None
        app.state.model_ready = False
        app.state.load_error = str(e)
        print(f"❌ Model loading failed: {e}")

@app.get("/health")
def health():
    if app.state.model_ready:
        validation_results = validate_model_performance(app.state.model_metadata)
        return {
            "status": "ok", 
            "model": True,
            "model_validation": validation_results,
            "model_accuracy": app.state.model_metadata.get("test_accuracy", "N/A") if app.state.model_metadata else "N/A"
        }
    else:
        return {
            "status": "degraded", 
            "model": False, 
            "error": getattr(app.state, "load_error", None)
        }

@app.get("/model/info", response_model=ModelInfo)
def get_model_information():
    """Get comprehensive information about the current model."""
    wandb_project = os.getenv("WANDB_PROJECT", "mlops-capstone")
    wandb_entity = os.getenv("WANDB_ENTITY", None)
    
    try:
        model_info = get_model_info(wandb_entity, wandb_project)
        return ModelInfo(**model_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch model info: {str(e)}")

@app.get("/model/validation")
def get_model_validation():
    """Get model validation results."""
    if not app.state.model_ready:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    validation_results = validate_model_performance(app.state.model_metadata)
    return {
        "validation_results": validation_results,
        "model_metadata": app.state.model_metadata
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if not app.state.model_ready:
        raise HTTPException(
            status_code=503, 
            detail=f"Model not ready: {getattr(app.state, 'load_error', 'Unknown error')}"
        )
    
    # Validate input
    if len(req.features) != 4:
        raise HTTPException(
            status_code=400, 
            detail="Expected exactly 4 features (sepal length, sepal width, petal length, petal width)"
        )
    
    try:
        features = np.array(req.features).reshape(1, -1)
        pred = model.predict(features).tolist()
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features).tolist()
        
        # Map prediction to class names if available
        class_names = app.state.model_metadata.get("target_classes", ["setosa", "versicolor", "virginica"])
        predicted_class = class_names[pred[0]] if pred[0] < len(class_names) else f"class_{pred[0]}"
        
        return {
            "prediction": pred,
            "predicted_class": predicted_class,
            "probability": proba,
            "model_info": {
                "accuracy": app.state.model_metadata.get("test_accuracy", "N/A"),
                "f1_score": app.state.model_metadata.get("test_f1", "N/A"),
                "cv_mean": app.state.model_metadata.get("cv_mean", "N/A")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "Iris Classification API",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "model_info": "/model/info",
            "model_validation": "/model/validation"
        },
        "model_ready": app.state.model_ready
    }
