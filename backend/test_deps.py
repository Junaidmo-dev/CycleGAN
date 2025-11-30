import sys
import os

print(f"Python executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")

try:
    import tqdm
    print(f"✅ tqdm imported successfully. File: {tqdm.__file__}")
except ImportError as e:
    print(f"❌ Failed to import tqdm: {e}")

try:
    import cv2
    print(f"✅ cv2 imported successfully. Version: {cv2.__version__}")
except ImportError as e:
    print(f"❌ Failed to import cv2: {e}")
