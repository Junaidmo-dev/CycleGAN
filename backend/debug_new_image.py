from ultralytics import YOLO
import sys

# Define comprehensive animal classes (Land, Air, Underwater)
ANIMAL_CLASSES = [
    # Underwater
    "fish", "jellyfish", "shark", "turtle", "sea turtle", "whale", "dolphin", 
    "starfish", "crab", "lobster", "octopus", "squid", "seal", "penguin", "ray", 
    "seahorse", "eel", "coral",
    
    # Land
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", 
    "lion", "tiger", "monkey", "rabbit", "deer", "fox", "wolf", "kangaroo", 
    "camel", "hippo", "rhino", "pig", "goat", "snake", "lizard", "frog",
    
    # Air
    "bird", "eagle", "parrot", "owl", "duck", "swan", "flamingo", "bat"
]

# The new image uploaded by user
image_path = "C:/Users/junai/.gemini/antigravity/brain/ec751dcd-0038-40b3-b975-18ae6fe97927/uploaded_image_1764424893104.png"

print(f"Testing YOLO-World on: {image_path}")

try:
    # Load model
    model = YOLO("yolov8s-world.pt")
    
    # Set classes
    model.set_classes(ANIMAL_CLASSES)
    
    # Run inference
    results = model.predict(image_path, conf=0.01) # Ultra low confidence
    
    print(f"\nResults found: {len(results)}")
    
    for r in results:
        print(f"Boxes found: {len(r.boxes)}")
        for box in r.boxes:
            cls = int(box.cls[0])
            if hasattr(model, 'names'):
                name = model.names[cls]
            else:
                name = f"Class {cls}"
                
            conf = float(box.conf[0])
            print(f" - Detected: {name} ({conf:.2f})")

except Exception as e:
    print(f"Error: {e}")
