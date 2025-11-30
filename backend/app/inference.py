import torch
from torchvision import transforms
from torchvision import transforms
from PIL import Image, ImageFilter
import io
from .model_loader import model_loader
from .utils import logger
import sys
import os
from torchvision import transforms
import numpy as np

# Add img2img_turbo to path
img2img_path = os.path.join(os.path.dirname(__file__), "..", "img2img_turbo")
sys.path.append(img2img_path)
logger.info(f"Added {img2img_path} to sys.path")

# Debug imports
try:
    import cv2
    logger.info(f"[SUCCESS] cv2 imported successfully. Version: {cv2.__version__}")
except ImportError as e:
    logger.error(f"[ERROR] Failed to import cv2: {e}")
    # Print sys.path to help debug
    logger.error(f"sys.path: {sys.path}")

def canny_from_pil_fallback(image, low_threshold=100, high_threshold=200):
    """Fallback edge detection using PIL if cv2 is missing"""
    logger.warning("Using PIL fallback for edge detection (suboptimal)")
    # Convert to grayscale
    image = image.convert("L")
    # Apply edge detection
    image = image.filter(ImageFilter.FIND_EDGES)
    # Convert back to RGB
    return image.convert("RGB")

try:
    from src.image_prep import canny_from_pil
    logger.info("[SUCCESS] canny_from_pil imported successfully")
except ImportError as e:
    canny_from_pil = canny_from_pil_fallback
    logger.warning(f"[WARNING] Could not import canny_from_pil: {e}. Using PIL fallback.")

def transform_image(image_bytes: bytes) -> torch.Tensor:
    """
    Convert raw bytes to a normalized tensor.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Resize to standard size if needed, or keep original
        # CycleGANs often work on 256x256 patches or full images
        transform = transforms.Compose([
            transforms.Resize((512, 512)), # Resize for consistency/speed in this demo
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1, 1]
        ])
        
        return transform(image).unsqueeze(0) # Add batch dimension
    except Exception as e:
        logger.error(f"Error transforming image: {e}")
        raise e

def tensor_to_image(tensor: torch.Tensor) -> bytes:
    """
    Convert output tensor back to image bytes.
    """
    try:
        # Denormalize: [-1, 1] -> [0, 1]
        tensor = tensor * 0.5 + 0.5
        tensor = torch.clamp(tensor, 0, 1)
        
        transform = transforms.ToPILImage()
        image = transform(tensor.squeeze(0).cpu())
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    except Exception as e:
        logger.error(f"Error converting tensor to image: {e}")
        raise e

def run_inference(image_bytes: bytes, model_name: str = 'raunenet') -> tuple[bytes, dict]:
    """
    Full inference pipeline: Bytes -> Tensor -> Model -> Tensor -> Bytes
    """
    model, device, model_type = model_loader.get_model(model_name)
    
    msg = f"🔄 Starting {model_type} inference..."
    print(msg)  # Force print to console
    logger.info(msg)
    
    logger.info(f"   Model type: {type(model).__name__}")
    logger.info(f"   Device: {device}")
    
    input_tensor = transform_image(image_bytes).to(device)
    logger.info(f"   Input tensor shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    msg_done = f"✅ {model_type} inference completed successfully"
    print(msg_done)  # Force print to console
    logger.info(msg_done)
    
    metadata = {
        "model_type": type(model).__name__,
        "device": str(device),
        "input_shape": str(input_tensor.shape),
        "execution_status": "success"
    }
    
    return tensor_to_image(output_tensor), metadata

def run_img2img_turbo(image_bytes: bytes, prompt: str, steps: int = 20, cfg: float = 7.5, skip_canny: bool = False, model_type: str = "edge_to_image", gamma: float = 0.4) -> bytes:
    """
    Runs the img2img-turbo diffusion pipeline.
    Supports 'edge_to_image' (default) and 'sketch_to_image_stochastic'.
    """
    # Get model from loader
    try:
        model_key = "img2img_turbo" if model_type == "edge_to_image" else "sketch_turbo_stochastic"
        model, device, _ = model_loader.get_model(model_key)
    except Exception as e:
        logger.error(f"Model not found: {e}")
        raise e

    # Convert input bytes -> PIL
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Ensure dimensions are multiples of 8 (required by the model)
    new_width = input_image.width - input_image.width % 8
    new_height = input_image.height - input_image.height % 8
    input_image = input_image.resize((new_width, new_height), Image.LANCZOS)

    noise = None
    if model_type == "sketch_to_image_stochastic":
        # Stochastic Sketch Mode
        # 1. Convert to grayscale to ensure clean thresholding
        gray_image = input_image.convert("L")
        
        # 2. Threshold: Lines (dark) become 1, Background (light) becomes 0
        # This creates a "White lines on Black background" map
        image_t = transforms.ToTensor()(gray_image) < 0.5
        
        # 3. Convert to float and normalize to [-1, 1]
        # 0 (Background) -> -1 (Black)
        # 1 (Lines) -> 1 (White)
        c_t = image_t.float() * 2 - 1
        
        # 4. Expand to 3 channels (VAE expects RGB) and add batch dim
        c_t = c_t.unsqueeze(0).repeat(1, 3, 1, 1).to(device)
        
        # Generate noise for stochastic variation
        B, C, H, W = c_t.shape
        noise = torch.randn((1, 4, H // 8, W // 8), device=device)
        
        logger.info(f"[STOCHASTIC MODE] Gamma: {gamma}, Input range: [{c_t.min()}, {c_t.max()}]")
    
    elif skip_canny:
        # Standard Sketch Mode (Deterministic)
        # 1. Convert to grayscale
        gray_image = input_image.convert("L")
        
        # 2. Threshold
        image_t = transforms.ToTensor()(gray_image) < 0.5
        
        # 3. Normalize to [-1, 1]
        c_t = image_t.float() * 2 - 1
        
        # 4. Expand to 3 channels
        c_t = c_t.unsqueeze(0).repeat(1, 3, 1, 1).to(device)
        
        logger.info("[SKETCH MODE] Using sketch preprocessing (threshold at 0.5, normalized to [-1, 1])")
    else:
        # Preprocess: Canny Edge Detection
        canny = canny_from_pil(input_image, 100, 200)
        c_t = transforms.ToTensor()(canny).unsqueeze(0).to(device)
        logger.info("[CANNY MODE] Using edge detection preprocessing")

    # Log tensor and model info
    logger.info(f"Input tensor - shape: {c_t.shape}, device: {c_t.device}, dtype: {c_t.dtype}")
    logger.info(f"Model device: {device}")
    logger.info(f"Prompt: '{prompt}'")

    # DO NOT modify the prompt - the model is trained to work with natural prompts
    # Quality boosters can actually interfere with the model's behavior
    logger.info(f"Using prompt: {prompt}")

    # Run inference
    with torch.no_grad():
        if model_type == "sketch_to_image_stochastic":
            output_image = model(c_t, prompt=prompt, deterministic=False, r=gamma, noise_map=noise)
        else:
            output_image = model(c_t, prompt=prompt)
    
    logger.info(f"✓ Inference complete - output shape: {output_image.shape}")
        
    # Postprocess (denormalize from [-1, 1] to [0, 1])
    output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

    # Back to bytes
    buf = io.BytesIO()
    output_pil.save(buf, format="PNG")
    return buf.getvalue()

