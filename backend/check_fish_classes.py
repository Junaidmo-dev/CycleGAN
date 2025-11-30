from ultralytics import YOLO
import os
import requests

MODEL_URL = "https://huggingface.co/akridge/yolo8-fish-detector-grayscale/resolve/main/yolov8n_fish_trained.pt"
MODEL_PATH = os.path.join("models", "yolov8n_fish.pt")

if not os.path.exists(MODEL_PATH):
    print("Downloading fish model...")
    response = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)

model = YOLO(MODEL_PATH)
print("Fish Model Classes:", model.names)
