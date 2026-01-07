import torch
import sys

print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("\n--- Why it might not be working ---")
    if "+cpu" in torch.__version__:
        print("You have the CPU-ONLY version of Torch installed.")
    else:
        print("Torch doesn't see your GPU drivers (CUDA).")
    print("\nRun this command to fix it:")
    print("pip uninstall torch torchvision -y && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
