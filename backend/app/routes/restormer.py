from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from ..realesrgan.service import realesrgan_service
from ..utils import logger

router = APIRouter()

@router.post("/restormer/enhance", summary="Enhance image using Real-ESRGAN")
async def enhance_with_realesrgan(file: UploadFile = File(...)):
    """
    Enhance an image using Real-ESRGAN (image restoration and upscaling).
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        logger.info(f"Processing image with Real-ESRGAN: {file.filename}")
        
        enhanced_bytes = realesrgan_service.process_image(contents)
        
        return Response(content=enhanced_bytes, media_type="image/png")
    except Exception as e:
        logger.error(f"Real-ESRGAN enhancement failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
