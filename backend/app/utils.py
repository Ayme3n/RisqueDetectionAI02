# utils.py
import joblib
import json
from pathlib import Path
from typing import Optional
import tensorflow as tf

# models directory (backend/models)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

def model_path(name: str) -> Path:
    return MODELS_DIR / name

def load_joblib(name: str):
    path = model_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Joblib artifact not found: {path}")
    return joblib.load(path)

def safe_load_joblib(name: str):
    """Return loaded joblib object or None if file missing."""
    path = model_path(name)
    if not path.exists():
        return None
    return joblib.load(path)

def load_json(name: str):
    path = model_path(name)
    if not path.exists():
        raise FileNotFoundError(f"JSON artifact not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def load_tf_model(name: str):
    """Load tensorflow saved model folder (returns None if missing)."""
    path = model_path(name)
    if not path.exists():
        return None
    # tf.keras.models.load_model accepts folder path for SavedModel or .h5 file
    return tf.keras.models.load_model(str(path))
