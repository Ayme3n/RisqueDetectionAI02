# inference.py
import os
from pathlib import Path
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from typing import Tuple, Optional

# local utils
from .utils import load_joblib, safe_load_joblib, load_json, load_tf_model

# ---- Globals (populated by load_artifacts) ----
scaler = None
iso_forest = None
oc_svm = None
autoencoder = None
encoded_columns = None
metadata = None
best_threshold = None
weights = None

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.array(v, dtype=float)
    if v.size == 0:
        return v
    mn = v.min()
    mx = v.max()
    if mx - mn <= 1e-12:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn)

def load_artifacts(models_path: Optional[str] = None):
    """
    Load artifacts from backend/models (default) or provided path.
    This populates module-level globals: scaler, iso_forest, oc_svm, autoencoder, encoded_columns, metadata, best_threshold, weights.
    Idempotent: calling multiple times is fine.
    """
    global scaler, iso_forest, oc_svm, autoencoder, encoded_columns, metadata, best_threshold, weights

    # if already loaded, skip
    if scaler is not None and encoded_columns is not None:
        return {
            "scaler": scaler,
            "iso_forest": iso_forest,
            "oc_svm": oc_svm,
            "autoencoder": autoencoder,
            "encoded_columns": encoded_columns,
            "metadata": metadata,
            "best_threshold": best_threshold,
            "weights": weights,
        }

    # allow override models_path for testing; otherwise utils uses default MODELS_DIR
    if models_path:
        # quick direct loads if custom path provided
        scaler = joblib.load(os.path.join(models_path, "scaler.joblib"))
        iso_forest = joblib.load(os.path.join(models_path, "iso_forest.joblib"))
        oc_svm = safe_load_joblib(os.path.join(models_path, "ocsvm.joblib"))  # may be None
        try:
            autoencoder = load_tf_model(os.path.join(models_path, "autoencoder.keras"))
        except Exception:
            autoencoder = None
        encoded_columns = joblib.load(os.path.join(models_path, "encoded_columns.pkl"))
        with open(os.path.join(models_path, "metadata.json")) as f:
            metadata = json.load(f)
    else:
        # normal path using helpers
        scaler = load_joblib("scaler.joblib")
        iso_forest = load_joblib("iso_forest.joblib")
        oc_svm = safe_load_joblib("ocsvm.joblib")
        autoencoder = load_tf_model("autoencoder.keras")
        encoded_columns = load_joblib("encoded_columns.pkl")
        metadata = load_json("metadata.json")

    best_threshold = metadata.get("best_threshold", 0.7)
    weights = metadata.get("weights", [0.4, 0.3, 0.3])

    # Normalize weights if some models are missing
    model_flags = [iso_forest is not None, autoencoder is not None, oc_svm is not None]
    if not all(model_flags):
        w = [weights[i] if model_flags[i] else 0.0 for i in range(3)]
        s = sum(w)
        if s == 0:
            # If none set (unlikely), fallback to iso only
            weights = [1.0, 0.0, 0.0]
        else:
            weights = [wi / s for wi in w]

    # Print short summary
    print("✅ Artifacts loaded:")
    print(f" - scaler: expecting {getattr(scaler,'n_features_in_', 'unknown')} features")
    print(f" - iso_forest: {'loaded' if iso_forest is not None else 'MISSING'}")
    print(f" - oc_svm: {'loaded' if oc_svm is not None else 'MISSING'}")
    print(f" - autoencoder: {'loaded' if autoencoder is not None else 'MISSING'}")
    print(f" - encoded_columns count: {len(encoded_columns) if encoded_columns is not None else 'MISSING'}")
    print(f" - best_threshold: {best_threshold}")
    print(f" - weights: {weights}")

    return {
        "scaler": scaler,
        "iso_forest": iso_forest,
        "oc_svm": oc_svm,
        "autoencoder": autoencoder,
        "encoded_columns": encoded_columns,
        "metadata": metadata,
        "best_threshold": best_threshold,
        "weights": weights,
    }

def preprocess(df: pd.DataFrame, encoded_cols=None) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Preprocess raw dataframe into the exact feature matrix expected by the scaler.
    Returns (X_scaled, df_encoded_aligned).
    If encoded_cols is None, will attempt to use loaded encoded_columns; if that is None, raises error.
    """
    global scaler, encoded_columns
    if encoded_cols is None:
        encoded_cols = encoded_columns

    if encoded_cols is None:
        raise ValueError("encoded_cols must be provided or load_artifacts() must be called first.")

    df_copy = df.copy()

    # Ensure numeric column exists and type-correct
    if "rule.level" in df_copy.columns:
        df_copy["rule.level"] = pd.to_numeric(df_copy["rule.level"], errors="coerce").fillna(0)

    # Ensure categorical columns exist
    categorical_cols = ["agent.name", "rule.mitre.id", "rule.mitre.tactic", "rule.mitre.technique"]
    for col in categorical_cols:
        if col not in df_copy.columns:
            df_copy[col] = None

    # Flatten list-like entries into first element (same logic as training)
    for col in ["rule.mitre.id", "rule.mitre.tactic", "rule.mitre.technique"]:
        df_copy[col] = df_copy[col].apply(lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else x)

    # One-hot encode
    df_encoded = pd.get_dummies(df_copy, columns=categorical_cols)

    # Align columns exactly to training encoded_cols:
    #  - add missing columns with zeros
    #  - drop any extras and enforce order
    for col in encoded_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded.reindex(columns=encoded_cols, fill_value=0)

    # Impute and scale
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(df_encoded)

    if scaler is None:
        raise RuntimeError("Scaler not loaded. Call load_artifacts() before preprocessing.")

    X_scaled = scaler.transform(X_imputed)

    return X_scaled, df_encoded

def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run inference on a raw dataframe (unencoded). Returns original dataframe + score columns.
    If artifacts aren't loaded yet, it will call load_artifacts() automatically.
    """
    global iso_forest, oc_svm, autoencoder, encoded_columns, best_threshold, weights

    if encoded_columns is None:
        # lazy-load artifacts if not already loaded
        load_artifacts()

    X, df_encoded = preprocess(df, encoded_columns)

    # Prepare arrays
    n = X.shape[0]
    iso_scores = np.zeros(n)
    svm_scores = np.zeros(n)
    ae_scores = np.zeros(n)

    if iso_forest is not None:
        iso_scores = -iso_forest.decision_function(X)
        iso_scores = _normalize(iso_scores)

    if oc_svm is not None:
        svm_scores = -oc_svm.decision_function(X)
        svm_scores = _normalize(svm_scores)

    if autoencoder is not None:
        preds = autoencoder.predict(X, verbose=0, batch_size=512)
        ae_raw = np.mean((X - preds) ** 2, axis=1)
        ae_scores = _normalize(ae_raw)

    # Ensemble
    ensemble_scores = weights[0] * iso_scores + weights[1] * ae_scores + weights[2] * svm_scores
    risky = (ensemble_scores > best_threshold).astype(int)

    # Return results attached to original raw DataFrame (keeps original columns)
    results = df.copy().reset_index(drop=True)
    results["iso_score"] = iso_scores
    results["svm_score"] = svm_scores
    results["ae_score"] = ae_scores
    results["ensemble_score"] = ensemble_scores
    results["risky"] = risky

    return results


if __name__ == "__main__":
    # quick manual run (from backend/app/)
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "cleaned_structured_wazuh_logs.csv"
    if csv_path.exists():
        load_artifacts()
        df = pd.read_csv(str(csv_path))
        res = predict_batch(df)
        print("Total logs:", len(res))
        print("Risky logs detected:", int(res["risky"].sum()))
        out_path = Path(__file__).resolve().parent.parent / "models" / "detected_risky_logs.csv"
        res[res["risky"] == 1].to_csv(out_path, index=False)
        print("Saved risky logs to", out_path)
    else:
        print("No sample CSV found at", csv_path)
