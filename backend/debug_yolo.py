from ultralytics import YOLO
from PIL import Image
import os

# Path to the image
image_path = "C:/Users/junai/.gemini/antigravity/brain/ec751dcd-0038-40b3-b975-18ae6fe97927/uploaded_image_1764423802807.png"
model_path = "yolov8n.pt"

print(f"Testing detection on: {image_path}")

try:
    # Load model
    model = YOLO(model_path)
    print("Model loaded successfully")

    # Run inference with low confidence to see EVERYTHING
    results = model(image_path, conf=0.1) 
    
    print(f"\nResults found: {len(results)}")
    
    for r in results:
        print(f"Boxes found: {len(r.boxes)}")
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0])
            print(f" - Detected: {name} ({conf:.2f})")
            
        # Save result image
        r.save(filename="debug_detection_result.jpg")
        print("\nSaved debug visualization to 'debug_detection_result.jpg'")

except Exception as e:
    print(f"Error: {e}")
