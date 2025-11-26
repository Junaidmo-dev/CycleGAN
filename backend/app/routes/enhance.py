from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import Response
from ..schemas import EnhancementResponse
from ..inference import run_inference
from ..utils import logger
from ..config import settings
from ..workers.tasks import enhance_image_task
import io
import base64

router = APIRouter()

@router.post("/enhance", summary="Enhance an image synchronously")
async def enhance_image(file: UploadFile = File(...)):
    """
    Upload an underwater image and get the enhanced version back immediately.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file size (approximate via seek/tell if needed, or read chunks)
    # For simplicity, we read and check length
    contents = await file.read()
    
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    try:
        print("🔍 enhance_image endpoint called!")
        logger.info(f"Processing image: {file.filename}")
        enhanced_bytes, metadata = run_inference(contents)
        
        # Add metadata to headers (must be strings)
        headers = {
            "X-Model-Type": metadata["model_type"],
            "X-Device": metadata["device"],
            "X-Input-Shape": metadata["input_shape"],
            "X-Execution-Status": metadata["execution_status"]
        }
        
        return Response(content=enhanced_bytes, media_type="image/png", headers=headers)
        
    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/enhance-async", summary="Enhance an image asynchronously")
async def enhance_image_async(file: UploadFile = File(...)):
    """
    Upload an image and get a job ID for background processing.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    
    if len(contents) > settings.MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
        
    try:
        # Encode to base64 for Celery transport
        image_b64 = base64.b64encode(contents).decode("utf-8")
        
        # Trigger Celery task
        task = enhance_image_task.delay(image_b64, file.filename)
        
        return {"job_id": task.id, "status": "submitted"}
        
    except Exception as e:
        logger.error(f"Async enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
