from pydantic import BaseModel
from typing import Optional

class EnhancementResponse(BaseModel):
    filename: str
    content_type: str
    size: int
    message: str

class JobResponse(BaseModel):
    job_id: str
    status: str
    result_url: Optional[str] = None
    error: Optional[str] = None

class HealthCheck(BaseModel):
    status: str
    version: str
