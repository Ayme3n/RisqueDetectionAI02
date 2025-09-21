# backend/tests/test_inference.py

import os
import pandas as pd
import numpy as np
import pytest

# Import inference pipeline
from backend.app import inference


@pytest.fixture(scope="session", autouse=True)
def setup_artifacts():
    """Load artifacts once before all tests."""
    inference.load_artifacts()


def test_artifacts_loaded():
    """Check that all required artifacts are loaded correctly."""
    assert inference.scaler is not None
    assert inference.encoded_columns is not None
    assert inference.iso_forest is not None
    assert inference.oc_svm is not None
    assert inference.autoencoder is not None
    assert inference.best_threshold is not None
    assert inference.weights is not None


def test_preprocess_feature_alignment():
    """Check that preprocess always aligns to the scaler feature count."""
    df = pd.DataFrame([{
        "rule.level": 5,
        "rule.id": 100001,
        "hour": 12,
        "dayofweek": 3,
        "agent.name": "agent-100",
        "rule.mitre.id": "T1003",
        "rule.mitre.tactic": "Credential Access",
        "rule.mitre.technique": "OS Credential Dumping"
    }])

    X, _ = inference.preprocess(df, inference.encoded_columns)
    assert X.shape[1] == inference.scaler.n_features_in_



def test_predict_batch_output():
    """Ensure predict_batch returns the expected dataframe with risky column."""
    df = pd.DataFrame([{
        "rule.level": 5,
        "rule.id": 100001,
        "hour": 12,
        "dayofweek": 3,
        "agent.name": "agent-100",
        "rule.mitre.id": "T1003",
        "rule.mitre.tactic": "Credential Access",
        "rule.mitre.technique": "OS Credential Dumping"
    }])

    results = inference.predict_batch(df)
    assert "risky" in results.columns
    assert results["risky"].isin([0, 1]).all()


def test_batch_consistency():
    """Ensure running on multiple logs produces consistent risky column (0/1)."""
    df = pd.DataFrame([{
        "rule.level": 10,
        "rule.id": 200001,
        "hour": 18,
        "dayofweek": 4,
        "agent.name": "agent-200",
        "rule.mitre.id": "T1059",
        "rule.mitre.tactic": "Execution",
        "rule.mitre.technique": "Command-Line Interface"
    } for _ in range(5)])

    results = inference.predict_batch(df)
    assert "risky" in results.columns
    assert set(results["risky"].unique()).issubset({0, 1})
