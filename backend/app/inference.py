import torch
from torchvision import transforms
from PIL import Image
import io
from .model_loader import model_loader
from .utils import logger

def transform_image(image_bytes: bytes) -> torch.Tensor:
    """
    Convert raw bytes to a normalized tensor.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Resize to standard size if needed, or keep original
        # CycleGANs often work on 256x256 patches or full images
        transform = transforms.Compose([
            transforms.Resize((256, 256)), # Resize for consistency/speed in this demo
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

def run_inference(image_bytes: bytes) -> tuple[bytes, dict]:
    """
    Full inference pipeline: Bytes -> Tensor -> Model -> Tensor -> Bytes
    """
    model, device, model_type = model_loader.get_model()
    
    msg = "🔄 Starting CycleGAN inference..."
    print(msg)  # Force print to console
    logger.info(msg)
    
    logger.info(f"   Model type: {type(model).__name__}")
    logger.info(f"   Device: {device}")
    
    input_tensor = transform_image(image_bytes).to(device)
    logger.info(f"   Input tensor shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
        
    msg_done = "✅ CycleGAN inference completed successfully"
    print(msg_done)  # Force print to console
    logger.info(msg_done)
    
    metadata = {
        "model_type": type(model).__name__,
        "device": str(device),
        "input_shape": str(input_tensor.shape),
        "execution_status": "success"
    }
    
    return tensor_to_image(output_tensor), metadata
