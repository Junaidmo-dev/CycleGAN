import sys
import os
import requests
from PIL import Image
import io

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "app"))

# Mock logger to avoid import errors if utils is complex
import logging
logging.basicConfig(level=logging.INFO)

try:
    from app.routes.detect import get_model, ANIMAL_CLASSES
    print("Successfully imported detect module")
except ImportError as e:
    print(f"Error importing detect module: {e}")
    sys.path.append(os.path.dirname(__file__))
    from app.routes.detect import get_model, ANIMAL_CLASSES

def verify_model():
    print("1. Initializing model (this should trigger download if missing)...")
    model = get_model()
    print("Model initialized successfully.")

    # Find the latest uploaded image (should be the stingray one)
    upload_dir = r"C:\Users\junai\.gemini\antigravity\brain\2e680882-a39b-44c0-b094-ef11a94183a2"
    images = [f for f in os.listdir(upload_dir) if f.endswith(".jpg") or f.endswith(".png")]
    
    if images:
        # Sort by modification time to get the absolute latest (dolphin)
        latest_image = max([os.path.join(upload_dir, f) for f in images], key=os.path.getmtime)
        print(f"2. Testing on image: {latest_image}")
        image = Image.open(latest_image)
    else:
        print("No images found in upload dir.")
        return
    
    print("3. Running inference...")
    # Ensure classes are set
    model.set_classes(ANIMAL_CLASSES)
    results = model(image, conf=0.05) # Low confidence
    
    print(f"4. Processing results...")
    for result in results:
        print(f"   Boxes found: {len(result.boxes)}")
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if hasattr(model, 'names'):
                label = model.names[cls]
            else:
                label = str(cls)
            print(f"   Detected: {label} ({conf:.2f})")
            
    print("Verification complete.")

if __name__ == "__main__":
    verify_model()
