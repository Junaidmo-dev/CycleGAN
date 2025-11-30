from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
from ultralytics import YOLO
from PIL import Image
import io
import requests
from ..utils import logger

router = APIRouter()

# Use YOLO-World (Open Vocabulary Detection)
# This allows us to detect ANY object by specifying text prompts
MODEL_NAME = "yolov8x-world.pt"  # Extra Large model for maximum accuracy

# Define comprehensive animal classes with descriptive prompts
ANIMAL_CLASSES = [
    # Underwater
    "jellyfish sea animal", "shark", "turtle", "sea turtle", "whale", "dolphin", 
    "starfish", "crab", "lobster", "octopus", "squid", "seal", "penguin", "ray", 
    "seahorse", "eel", "coral",
    
    # Land
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", 
    "lion", "tiger", "monkey", "rabbit", "deer", "mule deer", "white-tailed deer", "elk", "moose", "antelope", "fox", "wolf", "kangaroo", 
    "camel", "hippo", "rhino", "pig", "goat", "snake", "lizard", "frog",
    
    # Air
    "eagle", "parrot", "owl", "duck", "swan", "flamingo", "flying bat"
]

def get_model():
    """Load YOLO-World model and set custom classes"""
    try:
        logger.info(f"Loading YOLO-World model: {MODEL_NAME}")
        model = YOLO(MODEL_NAME)  # Auto-download
        # Force model onto CPU to avoid device mismatches
        model.to('cpu')
        
        # Set custom vocabulary for specific animal detection
        logger.info(f"Setting custom vocabulary with {len(ANIMAL_CLASSES)} animal classes...")
        model.set_classes(ANIMAL_CLASSES)
        
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Initialize model on module load (or lazy load)
# We'll lazy load to avoid startup delay if download is needed, 
# but for better UX, maybe trigger download on startup? 
# For now, lazy load in the endpoint or global variable.
model = None

@router.post("/detect")
async def detect_animals(file: UploadFile = File(...)):
    global model
    if model is None:
        model = get_model()

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Ensure classes are set (just in case)
        if hasattr(model, 'set_classes'):
            model.set_classes(ANIMAL_CLASSES)

        # Run detection with lower confidence
        results = model(image, conf=0.25)

        detections = []
        logger.info(f"Processing image... Found {len(results)} result objects")
        
        for result in results:
            logger.info(f"Result boxes: {len(result.boxes)}")
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Handle class names safely
                if hasattr(model, 'names'):
                    label = model.names[cls]
                else:
                    label = ANIMAL_CLASSES[cls] if cls < len(ANIMAL_CLASSES) else str(cls)
                
                logger.info(f"Detected: {label} ({conf:.2f})")
                
                detections.append({
                    "label": label,
                    "confidence": conf,
                    "box": [x1, y1, x2, y2]
                })

        return JSONResponse(content={"detections": detections})

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
