import pandas as pd
from backend.app import inference

# 1. Load the artifacts
artifacts = inference.load_artifacts()

# 2. Load your raw dataset
df = pd.read_csv("cleaned_structured_wazuh_logs.csv")

# 3. Run batch prediction
results = inference.predict_batch(df)

# 4. Summary
print("Total logs:", len(results))
print("Risky logs detected:", results["risky"].sum())
print("Normal logs:", (results["risky"] == 0).sum())

# 5. Save risky logs separately
risky_logs = results[results["risky"] == 1]
risky_logs.to_csv("detected_risky_logs.csv", index=False)
print(f"Saved {len(risky_logs)} risky logs to detected_risky_logs.csv")
