# inference.py
import os
import sys
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .utils import load_joblib, load_json, save_csv, safe_load_optional_joblib

# Globals (populated by load_artifacts)
scaler = None
iso_forest = None
oc_svm = None
autoencoder = None
encoded_columns = None
metadata = None
best_threshold = None
weights = None

# Default models folder is backend/models (utils builds that path)
def load_artifacts(models_path: str = None):
    """
    Load artifacts from backend/models. Call once at service startup.
    """
    global scaler, iso_forest, oc_svm, autoencoder, encoded_columns, metadata, best_threshold, weights

    # allow override, otherwise utils uses MODELS_DIR
    if models_path:
        # temporarily change utils MODELS_DIR by loading directly
        scaler = joblib.load(os.path.join(models_path, "scaler.joblib"))
        iso_forest = joblib.load(os.path.join(models_path, "iso_forest.joblib"))
        oc_svm = safe_load_optional_joblib(os.path.join(models_path, "ocsvm.joblib"))
        # try autoencoder folder or file
        ae_path = os.path.join(models_path, "autoencoder.keras")
        if os.path.exists(ae_path):
            autoencoder = tf.keras.models.load_model(ae_path)
        else:
            autoencoder = None
        encoded_columns = joblib.load(os.path.join(models_path, "encoded_columns.pkl"))
        with open(os.path.join(models_path, "metadata.json")) as f:
            metadata = json.load(f)
    else:
        # normal path via utils
        scaler = load_joblib("scaler.joblib")
        iso_forest = load_joblib("iso_forest.joblib")
        oc_svm = safe_load_optional_joblib("ocsvm.joblib")
        # autoencoder may be a folder saved by tf.keras.save
        models_dir = Path(__file__).resolve().parent.parent / "models"
        ae_path = models_dir / "autoencoder.keras"
        if ae_path.exists():
            autoencoder = tf.keras.models.load_model(str(ae_path))
        else:
            autoencoder = None
        encoded_columns = load_joblib("encoded_columns.pkl")
        metadata = load_json("metadata.json")

    best_threshold = metadata.get("best_threshold", 0.7)
    weights = metadata.get("weights", [0.4, 0.3, 0.3])

    # if oc_svm or autoencoder missing, normalize weights across available models
    available = [iso_forest is not None, autoencoder is not None, oc_svm is not None]
    # if any model is missing, renormalize the weights to sum to 1 for available ones
    if not all(available):
        w = weights.copy()
        # map to iso, ae, svm order
        model_flags = [iso_forest is not None, autoencoder is not None, oc_svm is not None]
        w = [w[i] if model_flags[i] else 0.0 for i in range(3)]
        s = sum(w)
        if s == 0:
            # fallback equal weights among available
            avail_count = sum(model_flags)
            if avail_count == 0:
                weights = [1.0, 0.0, 0.0]
            else:
                weights = [1.0/avail_count if model_flags[i] else 0.0 for i in range(3)]
        else:
            weights = [wi / s for wi in w]

    print("✅ Artifacts loaded:")
    print(f" - scaler: expecting {getattr(scaler,'n_features_in_', 'unknown')} features")
    print(f" - iso_forest: {'loaded' if iso_forest is not None else 'MISSING'}")
    print(f" - oc_svm: {'loaded' if oc_svm is not None else 'MISSING'}")
    print(f" - autoencoder: {'loaded' if autoencoder is not None else 'MISSING'}")
    print(f" - encoded_columns count: {len(encoded_columns) if encoded_columns is not None else 'MISSING'}")
    print(f" - best_threshold: {best_threshold}")
    print(f" - weights: {weights}")


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.array(v, dtype=float)
    if v.size == 0:
        return v
    mn = v.min()
    mx = v.max()
    if mx - mn <= 1e-12:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn)


def preprocess(df: pd.DataFrame, encoded_cols):
    """
    Preprocess raw dataframe into the exact feature matrix expected by the scaler.
    Steps:
     - ensure categorical columns exist
     - flatten list values where applicable
     - one-hot encode categorical cols
     - add missing columns from encoded_cols with zeros
     - drop extra columns and reorder exactly to encoded_cols
     - impute and return numpy array scaled by saved scaler
    """
    global scaler
    if encoded_cols is None:
        raise ValueError("encoded_cols must be provided (encoded_columns.pkl)")

    df_copy = df.copy()

    # Ensure numeric columns exist and correct types
    if "rule.level" in df_copy.columns:
        df_copy["rule.level"] = pd.to_numeric(df_copy["rule.level"], errors="coerce").fillna(0)

    # Ensure categorical columns exist (create with None if missing)
    categorical_cols = ["agent.name", "rule.mitre.id", "rule.mitre.tactic", "rule.mitre.technique"]
    for col in categorical_cols:
        if col not in df_copy.columns:
            df_copy[col] = None

    # Flatten list values to first element (same as training)
    for col in ["rule.mitre.id", "rule.mitre.tactic", "rule.mitre.technique"]:
        df_copy[col] = df_copy[col].apply(lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) > 0 else x)

    # One-hot encode categorical columns (this may create new columns)
    df_encoded = pd.get_dummies(df_copy, columns=categorical_cols)

    # Align with training encoded columns: add missing columns (zeros), drop extras, reorder
    for col in encoded_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    # If there are extras, keep only training columns (guarantee order)
    df_encoded = df_encoded.reindex(columns=encoded_cols, fill_value=0)

    # Impute missing numeric values
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(df_encoded)

    # Scale with saved scaler (expects exact number/order of cols)
    X_scaled = scaler.transform(X_imputed)

    return X_scaled, df_encoded


def predict_batch(df: pd.DataFrame, artifacts: dict):
    if artifacts is None:
        raise ValueError("Artifacts not loaded. Call load_artifacts() first.")
    """
    Run inference on a raw dataframe (unencoded).
    Requires load_artifacts() to have been called first.
    Returns a DataFrame with original columns + iso_score, svm_score, ae_score, ensemble_score, risky.
    """
    global encoded_columns, iso_forest, oc_svm, autoencoder, best_threshold, weights

    if encoded_columns is None:
        raise RuntimeError("Artifacts not loaded. Call load_artifacts() before predict_batch().")

    X, df_encoded = preprocess(df, encoded_columns)

    # Scores per model
    iso_scores = np.zeros(X.shape[0])
    svm_scores = np.zeros(X.shape[0])
    ae_scores = np.zeros(X.shape[0])

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

    # Ensemble combination
    ensemble_scores = weights[0] * iso_scores + weights[1] * ae_scores + weights[2] * svm_scores
    risky = (ensemble_scores > best_threshold).astype(int)

    # Build results DataFrame: include original raw df (not encoded) + scores/label
    results = df.copy().reset_index(drop=True)
    results["iso_score"] = iso_scores
    results["svm_score"] = svm_scores
    results["ae_score"] = ae_scores
    results["ensemble_score"] = ensemble_scores
    results["risky"] = risky

    return results


if __name__ == "__main__":
    # Quick local test (run from backend/app/)
    # expects cleaned_structured_wazuh_logs.csv in project root or adjust path.
    project_root = Path(__file__).resolve().parents[2]
    csv_path = project_root / "cleaned_structured_wazuh_logs.csv"
    if not csv_path.exists():
        print("Test CSV not found at", csv_path)
    else:
        load_artifacts()  # loads from backend/models by default
        df = pd.read_csv(str(csv_path))
        res = predict_batch(df)
        print("Total logs:", len(res))
        print("Risky logs detected:", int(res["risky"].sum()))
        out = Path(__file__).resolve().parent.parent / "models" / "detected_risky_logs.csv"
        res[res["risky"] == 1].to_csv(out, index=False)
        print("Saved risky logs to", out)
