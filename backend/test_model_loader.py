import sys
import os
import logging

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)

print("Importing model_loader...")
from app.model_loader import model_loader

print("Calling load_img2img_turbo()...")
model_loader.load_img2img_turbo()

if "img2img_turbo" in model_loader.models:
    print("✅ img2img_turbo loaded successfully!")
    print(f"Model type: {type(model_loader.models['img2img_turbo'])}")
else:
    print("❌ img2img_turbo failed to load.")
