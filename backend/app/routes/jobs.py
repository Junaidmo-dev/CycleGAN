from fastapi import APIRouter, HTTPException
from ..schemas import JobResponse
from celery.result import AsyncResult
from ..workers.celery_app import celery_app

router = APIRouter()

@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    """
    Get the status of a background enhancement job.
    """
    try:
        result = AsyncResult(job_id, app=celery_app)
        
        response = {
            "job_id": job_id,
            "status": result.status,
        }
        
        if result.ready():
            if result.successful():
                task_result = result.result
                if task_result.get("status") == "completed":
                    # Construct full URL for the result image
                    # In a real app, this would be a proper static file URL or S3 link
                    # For now, we return the path relative to storage
                    filename = task_result.get("filename")
                    response["result_url"] = f"/static/{filename}"
                else:
                    response["status"] = "FAILED"
                    response["error"] = task_result.get("error")
            else:
                response["status"] = "FAILED"
                response["error"] = str(result.result)
                
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
