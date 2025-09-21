import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ..app import inference

# Load models
inference.load_artifacts("backend/models")

# Test on some CSV logs (must be pre-encoded like training)
df = pd.read_csv("data/encoded_logs_with_labels.csv")

results = inference.predict_batch(df)
print(results.head())
