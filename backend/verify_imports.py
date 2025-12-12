print("Starting verification...")
try:
    import torch
    print(f"Torch imported: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"Failed to import torch: {e}")

try:
    import transformers
    print(f"Transformers imported: {transformers.__version__}")
except Exception as e:
    print(f"Failed to import transformers: {e}")

try:
    import accelerate
    print(f"Accelerate imported: {accelerate.__version__}")
except Exception as e:
    print(f"Failed to import accelerate: {e}")

try:
    import diffusers
    print(f"Diffusers imported: {diffusers.__version__}")
except Exception as e:
    print(f"Failed to import diffusers: {e}")

try:
    import peft
    print(f"Peft imported: {peft.__version__}")
except Exception as e:
    print(f"Failed to import peft: {e}")

from diffusers import StableDiffusionControlNetPipeline
print("StableDiffusionControlNetPipeline imported successfully!")
