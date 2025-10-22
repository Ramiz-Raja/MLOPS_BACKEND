# backend/app/model_manager.py
import os
import joblib
import pathlib
import wandb
import pandas as pd
from typing import Dict, Any, Optional, Tuple

MODEL_NAME = "iris-logreg-model"


def _normalize_entity_project(wandb_entity: Optional[str], wandb_project: Optional[str]) -> Tuple[Optional[str], str]:
    """
    Normalize W&B entity/project inputs and try to correct common misconfigurations.

    Handles cases where WANDB_PROJECT is mistakenly set to "entity/project" or includes extra slashes.
    """
    entity = (wandb_entity or "").strip("/") or None
    project = (wandb_project or "").strip("/")

    # If project accidentally contains a slash, try to split it
    if "/" in project:
        parts = [p for p in project.split("/") if p]
        if len(parts) == 2 and entity is None:
            # Project provided as "entity/project" and no entity env set
            entity, project = parts[0], parts[1]
        elif len(parts) >= 2 and entity:
            # Project mistakenly includes entity again (e.g., entity/project)
            if parts[0] == entity:
                project = parts[1]
            else:
                # Fall back to last segment as project name
                project = parts[-1]
        else:
            # Best-effort: take the last segment as the project
            project = parts[-1]

    if not project:
        raise ValueError("WANDB_PROJECT is required and cannot be empty")

    return entity, project


def _build_artifact_ref(
    wandb_entity: Optional[str],
    wandb_project: Optional[str],
    model_name: str,
    alias: str = "latest",
) -> str:
    """
    Build a valid W&B artifact reference in the form:
    [entity/]<project>/<artifact_name>:<alias>

    Also supports override via WANDB_ARTIFACT env var to use a full path directly.
    """
    # Highest priority: explicit override of full artifact ref
    override = os.getenv("WANDB_ARTIFACT") or os.getenv("MODEL_ARTIFACT")
    if override:
        return override.strip()

    entity, project = _normalize_entity_project(wandb_entity, wandb_project)

    if entity:
        return f"{entity}/{project}/{model_name}:{alias}"
    # No entity: use user default account context
    return f"{project}/{model_name}:{alias}"

def download_latest_model(wandb_entity: str, wandb_project: str, dest_dir: str = "/tmp/model") -> Tuple[str, Dict[str, Any]]:
    """
    Downloads the latest model artifact from W&B and returns local path to model file and metadata.
    
    Returns:
        Tuple of (model_file_path, artifact_metadata)
    """
    os.makedirs(dest_dir, exist_ok=True)
    api = wandb.Api()

    # Build a robust artifact reference
    artifact_ref = _build_artifact_ref(wandb_entity, wandb_project, MODEL_NAME, alias="latest")

    try:
        artifact = api.artifact(artifact_ref)
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch artifact {artifact_ref}: {e}. "
            "Check WANDB_ENTITY/WANDB_PROJECT values, or set WANDB_ARTIFACT to a full path like 'entity/project/artifact:alias'."
        )
    
    local_path = artifact.download(root=dest_dir)
    
    # Extract metadata from artifact
    metadata = artifact.metadata or {}
    # Add useful context
    metadata = {
        **metadata,
        "artifact_ref": artifact_ref,
        "artifact_name": getattr(artifact, "name", MODEL_NAME),
        "artifact_version": getattr(artifact, "version", None),
    }
    
    # Find model file
    model_file = None
    for p in pathlib.Path(local_path).rglob("*.joblib"):
        model_file = str(p)
        break
    
    # Fallback: try .pkl or model.joblib
    if not model_file:
        for p in pathlib.Path(local_path).rglob("*model*"):
            if p.suffix in (".pkl", ".joblib"):
                model_file = str(p)
                break
    
    if not model_file:
        raise FileNotFoundError("Model file not found in artifact")
    
    return model_file, metadata

def load_model_from_wandb(wandb_entity: str, wandb_project: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Loads the latest model from W&B along with its metadata.
    
    Returns:
        Tuple of (model, metadata)
    """
    model_file, metadata = download_latest_model(wandb_entity, wandb_project)
    model = joblib.load(model_file)
    return model, metadata

def get_model_info(wandb_entity: str, wandb_project: str) -> Dict[str, Any]:
    """
    Get comprehensive information about the latest model without loading it.
    
    Returns:
        Dictionary containing model metadata and performance metrics
    """
    try:
        api = wandb.Api()
        artifact_ref = _build_artifact_ref(wandb_entity, wandb_project, MODEL_NAME, alias="latest")
        
        artifact = api.artifact(artifact_ref)
        metadata = artifact.metadata or {}
        
        # Add artifact info
        info = {
            "artifact_ref": artifact_ref,
            "artifact_name": artifact.name,
            "artifact_version": artifact.version,
            "artifact_type": artifact.type,
            "artifact_description": artifact.description,
            "created_at": artifact.created_at.isoformat() if artifact.created_at else None,
            "updated_at": artifact.updated_at.isoformat() if artifact.updated_at else None,
            "metadata": metadata
        }
        
        return info
        
    except Exception as e:
        return {
            "error": f"Could not fetch model info: {str(e)}",
            "artifact_name": MODEL_NAME
        }

def validate_model_performance(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate model performance based on metadata.
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "model_loaded": True,
        "performance_metrics_available": False,
        "validation_passed": False,
        "warnings": [],
        "recommendations": []
    }
    
    if not metadata:
        validation_results["warnings"].append("No metadata available for model validation")
        return validation_results
    
    # Check if performance metrics are available
    performance_keys = ["test_accuracy", "test_f1", "cv_mean", "cv_std"]
    available_metrics = [key for key in performance_keys if key in metadata]
    
    if available_metrics:
        validation_results["performance_metrics_available"] = True
        
        # Check accuracy threshold
        if "test_accuracy" in metadata:
            accuracy = metadata["test_accuracy"]
            if accuracy >= 0.9:
                validation_results["recommendations"].append("Model accuracy is excellent (≥90%)")
            elif accuracy >= 0.8:
                validation_results["recommendations"].append("Model accuracy is good (≥80%)")
            else:
                validation_results["warnings"].append(f"Model accuracy is below 80%: {accuracy:.2%}")
        
        # Check cross-validation stability
        if "cv_std" in metadata:
            cv_std = metadata["cv_std"]
            if cv_std <= 0.05:
                validation_results["recommendations"].append("Model shows good stability (low CV variance)")
            else:
                validation_results["warnings"].append(f"Model shows high variance: {cv_std:.3f}")
        
        # Check for overfitting
        if "train_accuracy" in metadata and "test_accuracy" in metadata:
            train_acc = metadata["train_accuracy"]
            test_acc = metadata["test_accuracy"]
            diff = abs(train_acc - test_acc)
            if diff <= 0.1:
                validation_results["recommendations"].append("No significant overfitting detected")
            else:
                validation_results["warnings"].append(f"Potential overfitting: train-test gap = {diff:.3f}")
        
        # Overall validation
        validation_results["validation_passed"] = len(validation_results["warnings"]) == 0
    
    return validation_results
