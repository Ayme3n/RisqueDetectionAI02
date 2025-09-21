from fastapi import FastAPI, HTTPException
import pandas as pd
from backend.app import inference
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI()

class LogBatch(BaseModel):
    records: List[Dict[str, Any]]

@app.on_event("startup")
def startup_event():
    inference.load_artifacts()
    print("✅ Artifacts loaded at API startup")

@app.post("/predict")
def predict(batch: LogBatch):
    try:
        df = pd.DataFrame(batch.records)
        results = inference.predict_batch(df)
        return results.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
