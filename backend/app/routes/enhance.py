from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
import torch
from fastapi.responses import Response
from ..schemas import EnhancementResponse
from ..schemas import EnhancementResponse
from ..inference import run_inference
from ..utils import logger
from ..config import settings
from ..workers.tasks import enhance_image_task
import io
import base64

router = APIRouter()

@router.post("/enhance", summary="Enhance an image synchronously")
async def enhance_image(
    file: UploadFile = File(...),
    model: str = Form("raunenet"),
    prompt: str = Form(None),
    steps: int = Form(20),
    cfg: float = Form(7.5),
    skip_canny: bool = Form(False),
    gamma: float = Form(0.4)
):
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
        print(f"🔍 enhance_image endpoint called with model: {model}")
        print(f"📝 Prompt: {prompt}, Skip Canny: {skip_canny}")
        logger.info(f"Processing image: {file.filename} with model: {model}")
        

        if model == "controlnet":
            logger.info("Taking ControlNet path")
            if not prompt:
                raise HTTPException(status_code=400, detail="Prompt is required for ControlNet")
            
            from ..controlnet import controlnet_service
            # controlnet_service is a singleton, but we ensure it's loaded via model_loader conceptually or directly here
            # Ideally we should use model_loader to get it, but for simplicity we import the singleton
            # However, to respect the lazy loading pattern in model_loader, let's trigger it there first
            # But since we are inside the route, we can just use the service directly if it handles its own loading (which it does)
            
            enhanced_bytes = controlnet_service.process_image(
                contents,
                prompt,
                steps=steps,
                controlnet_conditioning_scale=gamma # Reusing gamma as conditioning scale for simplicity in UI
            )
            metadata = {
                "model_type": "ControlNet",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "input_shape": "dynamic",
                "execution_status": "success"
            }
        else:
            logger.info(f"Taking standard inference path for model: {model}")
            enhanced_bytes, metadata = run_inference(contents, model_name=model)
        
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
async def enhance_image_async(
    file: UploadFile = File(...),
    model: str = Form("raunenet")
):
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
        task = enhance_image_task.delay(image_b64, file.filename, model_name=model)
        
        return {"job_id": task.id, "status": "submitted"}
        
    except Exception as e:
        logger.error(f"Async enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
