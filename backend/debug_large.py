from ultralytics import YOLO
from PIL import Image

image_path = "C:/Users/junai/.gemini/antigravity/brain/ec751dcd-0038-40b3-b975-18ae6fe97927/uploaded_image_1764424893104.png"

# Simplified classes
SIMPLE_CLASSES = ["turtle", "sea turtle", "fish", "whale"]

print(f"Testing YOLO-World LARGE on: {image_path}")

try:
    # Use Large model
    model = YOLO("yolov8l-world.pt")
    model.set_classes(SIMPLE_CLASSES)
    
    results = model.predict(image_path, conf=0.01)
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]
            conf = float(box.conf[0])
            print(f" - Detected: {name} ({conf:.2f})")

except Exception as e:
    print(f"Error: {e}")
