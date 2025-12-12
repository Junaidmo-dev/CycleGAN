from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
from PIL import Image
import io
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..utils import logger

router = APIRouter()

# Moondream 2 Model ID
MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-08-26" 

model = None
tokenizer = None

# Version counter for forcing model reload on code changes
_MODEL_VERSION = 10  # Increment this to force reload

def get_model():
    """Load Moondream 2 model"""
    global model, tokenizer
    
    # Force reload if code changed
    if hasattr(get_model, '_loaded_version') and get_model._loaded_version != _MODEL_VERSION:
        logger.info(f"Code version changed ({get_model._loaded_version} -> {_MODEL_VERSION}), forcing model reload")
        model = None
        tokenizer = None
    
    try:
        logger.info(f"Loading Moondream 2 model: {MODEL_ID}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        try:
            # Enable CuDNN benchmark for faster fixed-size input processing
            if device == "cuda":
                torch.backends.cudnn.benchmark = True
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"🚀 GPU OPTIMIZATION ENABLED: Using {gpu_name} with CuDNN Benchmark")
            
            logger.info("Creating new model instance...")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID, 
                revision=REVISION,
                trust_remote_code=True,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
        except Exception as e:
            logger.warning(f"Error configuring GPU optimizations: {e}")

        if model is None: # If model loading failed in the inner try, re-raise
            raise Exception("Model failed to load after GPU optimization attempt.")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)

        # [FIXED] Transformers downgraded to 4.44.2. No patches needed.
        # Compatible version handles GenerationMixin and generation_config natively.
        
        get_model._loaded_version = _MODEL_VERSION
        
        logger.info("Moondream model loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load Moondream model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@router.post("/detect")
async def detect_animals(file: UploadFile = File(...)):
    global model, tokenizer
    if model is None:
        model, tokenizer = get_model()

    try:
        start_time = time.time()
        
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        read_time = time.time()
        logger.info(f"Image load time: {read_time - start_time:.4f}s")
        
        # prompt
        # prompt
        # Step 1: Identify Species
        species_prompt = "What animal or creature is in this image? Provide the specific species name if you can identify it, or the general type of animal. Be concise."
        
        # Using inference_mode for speed
        with torch.inference_mode():
            enc_start = time.time()
            enc_image = model.encode_image(image)
            enc_end = time.time()
            logger.info(f"Encoding time: {enc_end - enc_start:.4f}s")
            
            gen_start = time.time()
            # 1. Get Species
            species_response = model.answer_question(enc_image, species_prompt, tokenizer)
            
            # 2. Get Description
            desc_prompt = "Describe this animal in detail, including its physical characteristics and habitat."
            description_response = model.answer_question(enc_image, desc_prompt, tokenizer)
            
            gen_end = time.time()
            logger.info(f"Generation time: {gen_end - gen_start:.4f}s")

        logger.info(f"Total processing time: {time.time() - start_time:.4f}s")
        
        # Post-processing
        species = species_response.strip()
        description = description_response.strip()
        
        # Estimate confidence based on specificity
        confidence = 0.85 if len(species) > 3 and len(species) < 50 else 0.70
        if "cannot" in species.lower() or "unclear" in species.lower() or "not visible" in species.lower():
            confidence = 0.0
            species = "Not detected"

        logger.info(f"Detected: {species} ({confidence})")
        
        return JSONResponse(content={
            "description": description,
            "species": species,
            "confidence": confidence,
            "detections": [] # Keep for compatibility
        })

    except Exception as e:
        logger.error(f"Moondream inference failed: {e}")
        # Log stack trace for debugging
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
