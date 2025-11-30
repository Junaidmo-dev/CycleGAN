import sys
import os
import traceback

# Setup paths exactly as we intend to do in model_loader.py
backend_path = os.path.dirname(os.path.abspath(__file__))
img2img_base = os.path.join(backend_path, "img2img_turbo")
src_path = os.path.join(img2img_base, "src")

print(f"Adding to sys.path: {img2img_base}")
sys.path.append(img2img_base)

# The original code in pix2pix_turbo.py does `sys.path.append("src")` which is relative to CWD.
# Since we run from backend root, "src" doesn't exist.
# We need to add the absolute path to src so that `from model import ...` works.
print(f"Adding to sys.path: {src_path}")
sys.path.append(src_path)

print("\n--- Attempting to import Pix2Pix_Turbo ---")
try:
    from src.pix2pix_turbo import Pix2Pix_Turbo
    print("✅ Successfully imported Pix2Pix_Turbo")
except Exception as e:
    print(f"❌ Failed to import Pix2Pix_Turbo")
    traceback.print_exc()
