from pydantic import BaseModel
from typing import List, Dict, Any

class LogBatch(BaseModel):
    records: List[Dict[str, Any]]
