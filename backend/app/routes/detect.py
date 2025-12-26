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
_MODEL_VERSION = 11  # Increment this to force reload

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
        img_width, img_height = image.size
        read_time = time.time()
        logger.info(f"Image load time: {read_time - start_time:.4f}s")
        
        # OPTIMIZED: Single combined prompt for speed with clearer instructions
        combined_prompt = """Analyze this image and identify the main animal.
Return the result in this EXACT format:
SPECIES: <animal name>
DESCRIPTION: <2 sentences description>"""
        
        # Using inference_mode for speed
        with torch.inference_mode():
            enc_start = time.time()
            enc_image = model.encode_image(image)
            enc_end = time.time()
            logger.info(f"Encoding time: {enc_end - enc_start:.4f}s")
            
            gen_start = time.time()
            
            # Single combined query
            response = model.answer_question(enc_image, combined_prompt, tokenizer)
            
            gen_end = time.time()
            logger.info(f"Generation time: {gen_end - gen_start:.4f}s")
            logger.info(f"Raw Model Response: {response}") # Debug logging

        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time:.4f}s")
        
        # Parse combined response with robust fallbacks
        species = "Unknown"
        description = response.strip()
        
        lines = response.split('\n')
        
        # Strategy 1: Look for explicit "SPECIES:" tag
        for line in lines:
            if line.upper().startswith("SPECIES:"):
                species = line.split(":", 1)[1].strip()
                # Remove any trailing punctuation
                species = species.rstrip(".,")
                break
        
        # Strategy 2: If no tag, try to extract description from "DESCRIPTION:" tag
        desc_parts = response.split("DESCRIPTION:")
        if len(desc_parts) > 1:
            description = desc_parts[1].strip()
        
        # Strategy 3: Heuristic Fallback if SPECIES is still Unknown
        if species.lower() == "unknown" or species == "":
            # If the response starts with "This is a [Animal]", extract [Animal]
            first_sentence = response.split('.')[0].strip()
            lower_sentence = first_sentence.lower()
            
            common_starts = ["this is a ", "this is an ", "a ", "an ", "image of a ", "image of an "]
            for start in common_starts:
                if lower_sentence.startswith(start):
                    # Extract the next 2-3 words as the potential species
                    rest = first_sentence[len(start):]
                    words = rest.split()
                    if words:
                        # Take up to 3 words (e.g., "Great White Shark")
                        candidate = " ".join(words[:3]).rstrip(",.")
                        species = candidate.capitalize()
                        break
            
            # Final fallback: just use the first few words if it looks like a label
            if species == "Unknown" and len(first_sentence) < 30:
                 species = first_sentence.rstrip(",.")
        
        # Estimate confidence based on specificity
        confidence = 0.85 if len(species) > 3 and len(species) < 50 else 0.70
        if "cannot" in species.lower() or "unclear" in species.lower() or "not visible" in species.lower():
            confidence = 0.0
            species = "Not detected"

        logger.info(f"Detected: {species} ({confidence})")
        
        # Create a centered bounding box (since Moondream doesn't provide exact coordinates)
        # This gives a visual indicator similar to the reference image
        # Box covers ~70% of image, centered
        margin_x = int(img_width * 0.15)
        margin_y = int(img_height * 0.10)
        
        detections = []
        if confidence > 0:
            detections.append({
                "label": species,
                "confidence": confidence,
                "box": {
                    "x1": margin_x,
                    "y1": margin_y,
                    "x2": img_width - margin_x,
                    "y2": img_height - margin_y
                }
            })
        
        return JSONResponse(content={
            "description": description,
            "species": species,
            "confidence": confidence,
            "detections": detections,
            "processing_time": round(total_time, 2)
        })

    except Exception as e:
        logger.error(f"Moondream inference failed: {e}")
        # Log stack trace for debugging
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

