from ultralytics import YOLO
from PIL import Image

image_path = "C:/Users/junai/.gemini/antigravity/brain/ec751dcd-0038-40b3-b975-18ae6fe97927/uploaded_image_1764424893104.png"

# Check image
img = Image.open(image_path)
print(f"Image size: {img.size}")

# Simplified classes
SIMPLE_CLASSES = ["turtle", "sea turtle", "fish", "whale"]

print(f"Testing YOLO-World with simplified classes: {SIMPLE_CLASSES}")

try:
    model = YOLO("yolov8s-world.pt")
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
