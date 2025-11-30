import sys
import os
import traceback

# Simulate the EXACT path setup from model_loader.py
backend_path = os.path.dirname(os.path.abspath(__file__))
img2img_base = os.path.join(backend_path, "img2img_turbo")
src_path = os.path.join(img2img_base, "src")

print(f"Backend path: {backend_path}")
print(f"img2img_base: {img2img_base}")
print(f"src_path: {src_path}")
print(f"img2img_base exists: {os.path.exists(img2img_base)}")
print(f"src_path exists: {os.path.exists(src_path)}")

sys.path.append(img2img_base)
sys.path.append(src_path)

print("\n--- Attempting to import Pix2Pix_Turbo ---")
try:
    from src.pix2pix_turbo import Pix2Pix_Turbo
    print("✅ Successfully imported Pix2Pix_Turbo")
    print(f"Pix2Pix_Turbo class: {Pix2Pix_Turbo}")
except Exception as e:
    print(f"❌ Failed to import Pix2Pix_Turbo")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
