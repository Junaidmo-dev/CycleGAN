import os
import sys
import shutil
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Define paths
UPLOAD_DIR = r"C:\Users\junai\.gemini\antigravity\brain\2e680882-a39b-44c0-b094-ef11a94183a2"
BACKEND_DIR = os.path.dirname(__file__)
DATASET_DIR = os.path.join(BACKEND_DIR, "dataset")
IMAGES_DIR = os.path.join(DATASET_DIR, "images", "train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels", "train")
VISUAL_DIR = os.path.join(UPLOAD_DIR, "visualized_detections")

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

# Add backend to path to import config if needed, but we'll just use the model directly
sys.path.append(os.path.join(BACKEND_DIR, "app"))

# Import class list from detect.py to ensure consistency
# Fallback if import fails OR if we want to force the base model for labeling
# We MUST use the base world model to detect new classes like 'lionfish'
ANIMAL_CLASSES = ["fish", "jellyfish", "shark", "turtle", "whale", "dolphin", "starfish", "stingray", "lionfish"]
MODEL_NAME = "yolov8x-worldv2.pt"

def process_images():
    print(f"Loading model: {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)
    model.set_classes(ANIMAL_CLASSES)
    
    # Find uploaded images
    image_files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith("uploaded_image_") and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images to process.")
    
    for img_file in image_files:
        src_path = os.path.join(UPLOAD_DIR, img_file)
        
        # 1. Run Detection
        print(f"Processing {img_file}...")
        results = model(src_path, conf=0.10) # Low confidence to catch everything for "training" data
        
        # 2. Save Image to Dataset
        # Rename to simple index or keep original
        dst_img_path = os.path.join(IMAGES_DIR, img_file)
        shutil.copy(src_path, dst_img_path)
        
        # 3. Generate Labels and Visualization
        label_path = os.path.join(LABELS_DIR, os.path.splitext(img_file)[0] + ".txt")
        
        # Open image for drawing
        im = Image.open(src_path)
        draw = ImageDraw.Draw(im)
        
        with open(label_path, "w") as f:
            for result in results:
                for box in result.boxes:
                    # YOLO format: class x_center y_center width height (normalized)
                    cls = int(box.cls[0])
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    f.write(f"{cls} {x_center} {y_center} {width} {height}\n")
                    
                    # Visualization
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    label_name = ANIMAL_CLASSES[cls] if cls < len(ANIMAL_CLASSES) else str(cls)
                    
                    # Draw box
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                    draw.text((x1, y1), f"{label_name} {conf:.2f}", fill="red")
        
        # Save visualization
        vis_path = os.path.join(VISUAL_DIR, "vis_" + img_file)
        im.save(vis_path)
        print(f"  Saved visualization to {vis_path}")
        print(f"  Saved labels to {label_path}")

    print("Processing complete.")
    print(f"Dataset prepared at {DATASET_DIR}")
    print(f"Visualizations at {VISUAL_DIR}")

if __name__ == "__main__":
    process_images()
