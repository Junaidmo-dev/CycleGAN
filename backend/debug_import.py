import sys
import os
import traceback

print(f"Current Working Directory: {os.getcwd()}")
print(f"Script Location: {os.path.abspath(__file__)}")

# Simulate the path modification in inference.py
# inference.py is in backend/app/
# We are running this from backend/ (likely) or root.
# Let's try to match inference.py's logic relative to THIS file if we place it in backend/

# If this script is in backend/debug_import.py
# inference.py is in backend/app/inference.py
# The path addition in inference.py is:
# sys.path.append(os.path.join(os.path.dirname(__file__), "..", "img2img_turbo"))

# So if we are in backend/, we want to add ./img2img_turbo
img2img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img2img_turbo")
print(f"Adding path: {img2img_path}")
sys.path.append(img2img_path)

print("Sys.path:")
for p in sys.path:
    print(f"  - {p}")

print("\n--- Attempting imports ---")

print("1. Importing cv2...")
try:
    import cv2
    print(f"   ✅ cv2 imported. Version: {cv2.__version__}")
except ImportError:
    print("   ❌ Failed to import cv2")
    traceback.print_exc()

print("\n2. Importing src.image_prep...")
try:
    import src.image_prep
    print("   ✅ src.image_prep imported")
except ImportError:
    print("   ❌ Failed to import src.image_prep")
    traceback.print_exc()

print("\n3. Importing canny_from_pil from src.image_prep...")
try:
    from src.image_prep import canny_from_pil
    print("   ✅ canny_from_pil imported")
except ImportError:
    print("   ❌ Failed to import canny_from_pil")
    traceback.print_exc()
