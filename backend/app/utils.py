# utils.py
import joblib
import json
from pathlib import Path
import pandas as pd

# Models directory (relative to backend/app => backend/models)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

def model_path(name: str) -> Path:
    return MODELS_DIR / name

def load_joblib(name: str):
    path = model_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Joblib artifact not found: {path}")
    return joblib.load(path)

def load_json(name: str):
    path = model_path(name)
    if not path.exists():
        raise FileNotFoundError(f"JSON artifact not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

def save_csv(df: pd.DataFrame, name: str):
    out = model_path(name)
    df.to_csv(out, index=False)
    return out

def safe_load_optional_joblib(name: str):
    """Load optional joblib artifact; return None if missing."""
    path = model_path(name)
    if not path.exists():
        return None
    return joblib.load(path)
