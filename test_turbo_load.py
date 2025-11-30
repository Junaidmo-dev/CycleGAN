
import sys
import os
import torch

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

# Add img2img_turbo to path
img2img_path = os.path.join(os.getcwd(), "backend", "img2img_turbo")
sys.path.append(img2img_path)
sys.path.append(os.path.join(img2img_path, "src"))

print(f"Python executable: {sys.executable}")
print(f"Cuda available: {torch.cuda.is_available()}")

try:
    from src.pix2pix_turbo import Pix2Pix_Turbo
    print("Successfully imported Pix2Pix_Turbo class")
    
    model = Pix2Pix_Turbo("edge_to_image")
    print("Successfully initialized Pix2Pix_Turbo model")
except Exception as e:
    print(f"Failed to load Pix2Pix_Turbo: {e}")
    import traceback
    traceback.print_exc()
