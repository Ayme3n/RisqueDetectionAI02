# main.py
from fastapi import FastAPI, HTTPException
import pandas as pd
from backend.app import inference
from backend.app.schemas import LogBatch

app = FastAPI(title="RisqueDetectionAI Inference API", version="1.0")

@app.on_event("startup")
def startup_event():
    # Load artifacts once at startup (inference will also lazy-load if missing)
    inference.load_artifacts()
    print("✅ Artifacts loaded at API startup")

@app.post("/predict")
def predict(batch: LogBatch):
    try:
        df = pd.DataFrame(batch.records)
        results = inference.predict_batch(df)
        # convert DataFrame to list-of-dicts for JSON response
        return {
            "status": "success",
            "num_records": len(results),
            "predictions": results.to_dict(orient="records")
        }
    except Exception as e:
        # Keep error message readable for debugging
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
