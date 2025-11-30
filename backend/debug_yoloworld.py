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

image_path = "C:/Users/junai/.gemini/antigravity/brain/ec751dcd-0038-40b3-b975-18ae6fe97927/uploaded_image_1764424614855.png"

print(f"Testing YOLO-World on: {image_path}")

try:
    # Load model
    model = YOLO("yolov8s-world.pt")
    print("Model loaded.")
    
    # Set classes
    print("Setting classes...")
    model.set_classes(ANIMAL_CLASSES)
    
    # Run inference
    print("Running inference...")
    results = model.predict(image_path, conf=0.05) # Very low confidence
    
    print(f"\nResults found: {len(results)}")
    
    for r in results:
        print(f"Boxes found: {len(r.boxes)}")
        for box in r.boxes:
            cls = int(box.cls[0])
            # For YOLO-World, names might be dynamic
            if hasattr(model, 'names'):
                name = model.names[cls]
            else:
                name = f"Class {cls}"
                
            conf = float(box.conf[0])
            print(f" - Detected: {name} ({conf:.2f})")
            
        r.save(filename="debug_yoloworld_result.jpg")
        print("Saved debug image.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
