import torch
from PIL import Image
import numpy as np
import io
import cv2

# Fix for torchvision compatibility with basicsr
import sys
import torchvision.transforms.functional as F_vision
sys.modules['torchvision.transforms.functional_tensor'] = F_vision

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
from ..utils import logger

class RealESRGANService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.upsampler = None

    def load_model(self):
        if self.upsampler is not None:
            return

        logger.info("Loading Real-ESRGAN model...")
        
        # Use RealESRGAN_x2plus (lightweight, good balance)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        
        self.upsampler = RealESRGANer(
            scale=2,
            model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
            model=model,
            tile=256,  # Use tiling for 4GB VRAM
            tile_pad=10,
            pre_pad=0,
            half=True,  # FP16 for memory efficiency
            device=self.device
        )
        
        logger.info("Real-ESRGAN loaded successfully.")

    def process_image(self, image_bytes: bytes) -> bytes:
        if self.upsampler is None:
            self.load_model()

        try:
            # 1. Load image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(image)
            
            # 2. Convert RGB to BGR for cv2
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # 3. Enhance
            with torch.no_grad():
                output, _ = self.upsampler.enhance(img_bgr, outscale=2)

            # 4. Convert BGR back to RGB
            output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            
            # 5. Convert to PIL
            output_image = Image.fromarray(output_rgb)

            # 6. Return bytes
            img_byte_arr = io.BytesIO()
            output_image.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()

        except Exception as e:
            logger.error(f"Real-ESRGAN inference failed: {e}")
            torch.cuda.empty_cache()
            raise e

# Singleton instance
realesrgan_service = RealESRGANService()
