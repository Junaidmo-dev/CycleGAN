import torch
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
import io
import os
from .model import Restormer
from ..utils import logger

class RestormerService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_path = os.path.join("models", "real_denoising.pth")

    def load_model(self):
        if self.model is not None:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Restormer weights not found at {self.model_path}. Please download them.")

        logger.info(f"Loading Restormer from {self.model_path}...")
        
        # Initialize model architecture
        self.model = Restormer(
            num_blocks=[4, 6, 6, 8],
            heads=[1, 2, 4, 8],
            dim=48,
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias'
        )

        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle state dict keys
        if 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Fix for potential key mismatch (remove 'module.' prefix if present)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '') if 'module.' in k else k
            new_state_dict[name] = v

        self.model.load_state_dict(new_state_dict, strict=False) # Allow loose loading for minor mismatches

        self.model.to(self.device)
        self.model.eval()
        logger.info("Restormer loaded successfully.")

    def process_image(self, image_bytes: bytes) -> bytes:
        if self.model is None:
            self.load_model()

        try:
            # 1. Load image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # 2. Preprocess
            # Convert to tensor and normalize to [0, 1]
            img_tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)

            # 3. Inference
            # Use FP16 (autocast) to save VRAM
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # Pad image to be multiple of 8 if needed (Restormer usually handles odd sizes but padding is safer)
                    _, _, h, w = img_tensor.shape
                    pad_h = (8 - h % 8) % 8
                    pad_w = (8 - w % 8) % 8
                    if pad_h != 0 or pad_w != 0:
                        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')

                    output_tensor = self.model(img_tensor)

                    # Unpad
                    output_tensor = output_tensor[:, :, :h, :w]

            # 4. Postprocess
            output_tensor = torch.clamp(output_tensor, 0, 1)
            output_image = TF.to_pil_image(output_tensor.squeeze(0).cpu())

            # 5. Return bytes
            img_byte_arr = io.BytesIO()
            output_image.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()

        except Exception as e:
            logger.error(f"Restormer inference failed: {e}")
            # Clean up VRAM on error
            torch.cuda.empty_cache()
            raise e

# Singleton instance
restormer_service = RestormerService()
