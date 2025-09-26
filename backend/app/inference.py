# backend/app/inference.py
import os
from pathlib import Path
from typing import Optional, Tuple

import joblib
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# local helpers (expects backend/app/utils.py to exist)
from .utils import load_joblib, safe_load_joblib, load_json, load_tf_model

# ---------------- Globals ----------------
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


# ---------------- Artifact loading ----------------
def load_artifacts(models_path: Optional[str] = None) -> dict:
    """
    Load artifacts from backend/models (default) or from models_path.
    Populates module-level globals and returns a dict of artifacts.
    Idempotent: calling multiple times is OK.
    """
    global scaler, iso_forest, oc_svm, autoencoder, encoded_columns, metadata, best_threshold, weights

    # If already loaded, return current objects
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

    # Allow overriding the models folder for testing
    if models_path:
        mp = Path(models_path)
        scaler = joblib.load(mp / "scaler.joblib")
        iso_forest = joblib.load(mp / "iso_forest.joblib")
        oc_svm = safe_load_joblib(str(mp / "ocsvm.joblib"))
        try:
            autoencoder = load_tf_model(str(mp / "autoencoder.keras"))
        except Exception:
            autoencoder = None
        encoded_columns = joblib.load(mp / "encoded_columns.pkl")
        with open(mp / "metadata.json", "r") as f:
            metadata = json.load(f)
    else:
        # use helpers that know backend/models location
        scaler = load_joblib("scaler.joblib")
        iso_forest = load_joblib("iso_forest.joblib")
        oc_svm = safe_load_joblib("ocsvm.joblib")
        autoencoder = load_tf_model("autoencoder.keras")
        encoded_columns = load_joblib("encoded_columns.pkl")
        metadata = load_json("metadata.json")

    best_threshold = metadata.get("best_threshold", 0.7)
    weights = metadata.get("weights", [0.4, 0.3, 0.3])

    # If any model missing, renormalize weights across available models
    model_flags = [iso_forest is not None, autoencoder is not None, oc_svm is not None]
    if not all(model_flags):
        w = [weights[i] if model_flags[i] else 0.0 for i in range(3)]
        s = sum(w)
        if s == 0:
            # fallback to using isolation forest only
            weights = [1.0, 0.0, 0.0]
        else:
            weights = [wi / s for wi in w]

    print("✅ Artifacts loaded:")
    print(f" - scaler: expecting {getattr(scaler, 'n_features_in_', 'unknown')} features")
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


# ---------------- Schema normalizer ----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize incoming DataFrame column names and ensure expected base columns.
    - rename legacy names (mitre.id -> rule.mitre.id, etc.)
    - drop free-text / redundant columns that were NOT used in training
    - ensure the column order (keeps the base set expected prior to one-hot)
    """
    df = df.copy()

    # Standardize some legacy column names if present
    rename_map = {
        "mitre.id": "rule.mitre.id",
        "mitre.tactic": "rule.mitre.tactic",
    }
    df = df.rename(columns=rename_map)

    # Drop free-text / redundant columns that were not used for encoding/training
    # You told me these were excluded during training:
    df = df.drop(columns=["rule.description", "rule.mitre.technique"], errors="ignore")

    # Ensure base columns are present (fill missing with None so encoding creates dummies)
    base_cols = [
        "timestamp",
        "agent.name",
        "rule.mitre.id",
        "rule.mitre.tactic",
        "rule.level",
        "rule.id",
    ]
    for c in base_cols:
        if c not in df.columns:
            df[c] = None

    # Keep only these base columns (training one-hot happens later)
    return df[base_cols]


# ---------------- Preprocessing ----------------
def preprocess(df: pd.DataFrame, encoded_cols: Optional[list] = None) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Preprocess raw input dataframe to the numeric matrix expected by the scaler.
    Returns (X_scaled, df_encoded_aligned).
    Steps:
      - Normalize columns (names / drop unused text)
      - Flatten list-like mitre columns (take first element)
      - One-hot encode categorical columns used in training
      - Add missing training dummies with zeros, drop extras, enforce order
      - Fill NaNs, impute numeric missing, scale with saved scaler
    """
    global scaler, encoded_columns

    # Lazy load encoded_columns if not loaded yet
    if encoded_cols is None:
        if encoded_columns is None:
            load_artifacts()   # populates encoded_columns and scaler
        encoded_cols = encoded_columns

    # Normalize names & drop unused text cols
    df_norm = normalize_columns(df)

    # Ensure categorical columns exist
    categorical_cols = ["agent.name", "rule.mitre.id", "rule.mitre.tactic"]
    for col in categorical_cols:
        if col not in df_norm.columns:
            df_norm[col] = None

    # Flatten list-like cells to first element (same as training notebook)
    for col in ["rule.mitre.id", "rule.mitre.tactic"]:
        df_norm[col] = df_norm[col].apply(lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else x)

    # One-hot encode categorical columns (this may produce new columns)
    df_encoded = pd.get_dummies(df_norm, columns=categorical_cols)

    # Align with training encoded columns: add missing columns (zeros), drop extras and reorder
    for col in encoded_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    # Ensure exact ordering and drop any extras
    df_encoded = df_encoded.reindex(columns=encoded_cols, fill_value=0)

    # Replace any remaining NaNs (safety) with 0
    df_encoded = df_encoded.fillna(0)

    # Impute numeric missing values (keeps behavior similar to training)
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(df_encoded)

    # Ensure scaler is loaded
    if scaler is None:
        load_artifacts()

    # Scale
    X_scaled = scaler.transform(X_imputed)

    return X_scaled, df_encoded


# ---------------- Prediction ----------------

def _ensure_timestamp_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee there's a 'timestamp' column if any date/time-like column exists.
    - If 'timestamp' present: do nothing.
    - Else look for columns containing 'time' or 'date' and rename the first found to 'timestamp'.
    Returns modified copy (does not mutate original).
    """
    df2 = df.copy()
    if "timestamp" in df2.columns:
        return df2

    # heuristic: find any column with 'time' or 'date' in its name
    for c in df2.columns:
        low = c.lower()
        if "time" in low or "date" in low:
            df2 = df2.rename(columns={c: "timestamp"})
            return df2

    # nothing found — return original copy unchanged
    return df2


def _make_dataframe_json_safe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pandas DataFrame to a version safe for JSON serialization:
    - Replace NaN / inf with None
    - Convert numpy scalar types to python native types
    """
    df2 = df.copy(deep=True)

    # replace inf and -inf with NaN first, then nan -> None
    df2 = df2.replace([np.inf, -np.inf], np.nan)
    df2 = df2.where(pd.notnull(df2), None)

    # convert numpy scalars -> python scalars to avoid json encoding issues
    for col in df2.columns:
        # apply only when necessary
        if df2[col].dtype == object:
            # object columns can already contain strings / None — keep them
            df2[col] = df2[col].apply(lambda x: x.item() if isinstance(x, (np.generic,)) else x)
        else:
            # numeric columns: convert each element to python type but keep None
            df2[col] = df2[col].apply(lambda x: (x.item() if isinstance(x, (np.generic,)) else x) if x is not None else None)

    return df2

# --- new robust predict_batch ---
def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust batch prediction:
     - Lazy-load artifacts if needed
     - Preserve original input columns (timestamp retained)
     - Add parsed timestamp column: 'timestamp_parsed'
     - Compute iso_score, svm_score, ae_score, ensemble_score, risky
     - Return a JSON-safe DataFrame (NaN -> None, numpy scalars -> python scalars)
    """
    global iso_forest, oc_svm, autoencoder, encoded_columns, best_threshold, weights, scaler

    # Ensure artifacts are loaded
    if encoded_columns is None or scaler is None or iso_forest is None:
        load_artifacts()

    # Keep original input intact (copy)
    raw = df.copy().reset_index(drop=True)

    # Ensure we have a timestamp column if one exists under a different name
    raw = _ensure_timestamp_col(raw)

    # Create parsed timestamp column for plotting convenience (keeps original untouched)
    try:
        raw["timestamp_parsed"] = pd.to_datetime(raw["timestamp"], errors="coerce")
    except Exception:
        raw["timestamp_parsed"] = None

    # Preprocess -> get numeric feature matrix X (and df_encoded if your preprocess returns it)
    # Your preprocess signature previously returned (X, df_encoded). It should still work.
    X, _ = preprocess(raw, encoded_columns)  # preprocess must align features but not mutate `raw`

    n = X.shape[0]
    iso_scores = np.zeros(n)
    svm_scores = np.zeros(n)
    ae_scores = np.zeros(n)

    # Isolation Forest
    if iso_forest is not None:
        iso_scores = -iso_forest.decision_function(X)
        # normalize to 0..1
        iso_scores = _normalize(iso_scores)

    # One-Class SVM
    if oc_svm is not None:
        try:
            svm_scores = -oc_svm.decision_function(X)
            svm_scores = _normalize(svm_scores)
        except Exception:
            # fallback: zero vector
            svm_scores = np.zeros(n)

    # Autoencoder
    if autoencoder is not None:
        try:
            preds = autoencoder.predict(X, verbose=0, batch_size=512)
            ae_raw = np.mean((X - preds) ** 2, axis=1)
            ae_scores = _normalize(ae_raw)
        except Exception:
            ae_scores = np.zeros(n)

    # Ensemble (weights were normalized at load_artifacts())
    ensemble_scores = weights[0] * iso_scores + weights[1] * ae_scores + weights[2] * svm_scores
    risky = (ensemble_scores > best_threshold).astype(int)


    # Build results while preserving original columns (including timestamp)
    results = raw.copy().reset_index(drop=True)
    results["iso_score"] = iso_scores
    results["svm_score"] = svm_scores
    results["ae_score"] = ae_scores
    results["ensemble_score"] = ensemble_scores
    results["risky"] = risky

    
    # Make the DataFrame JSON safe BEFORE returning (important for FastAPI -> JSON)
    results_safe = _make_dataframe_json_safe(results)

    return results_safe


# ---------------- Quick CLI test when module run directly ----------------
if __name__ == "__main__":
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
