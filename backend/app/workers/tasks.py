from .celery_app import celery_app
from ..inference import run_inference
from ..config import settings
import os
import base64

@celery_app.task(bind=True, name="enhance_image_task")
def enhance_image_task(self, image_data_b64: str, filename: str, model_name: str = 'raunenet'):
    """
    Celery task to enhance an image in the background.
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(image_data_b64)
        
        # Run inference
        enhanced_bytes, _ = run_inference(image_bytes, model_name=model_name)
        
        # Save to storage
        output_filename = f"enhanced_{self.request.id}_{filename}"
        output_path = settings.STORAGE_PATH / output_filename
        
        with open(output_path, "wb") as f:
            f.write(enhanced_bytes)
            
        return {
            "status": "completed",
            "filename": output_filename,
            "path": str(output_path)
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }
