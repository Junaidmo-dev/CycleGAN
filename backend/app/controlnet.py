import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from .utils import logger
from .config import settings
import os
import io

class ControlNetService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        self.controlnet = None

    def load_model(self):
        if self.pipeline is not None:
            return

        try:
            logger.info("Loading ControlNet (Canny)...")
            
            # Load ControlNet
            controlnet_model_id = "lllyasviel/sd-controlnet-canny"
            # You might want to cache these locally or use a local path if available
            self.controlnet = ControlNetModel.from_pretrained(
                controlnet_model_id, 
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
            )

            # Load Stable Diffusion
            base_model_id = "runwayml/stable-diffusion-v1-5"
            self.pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                base_model_id,
                controlnet=self.controlnet,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                safety_checker=None # Optional: Disable for speed/memory if safe
            )

            # Scheduler
            self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
            
            # Optimizations
            if self.device.type == "cuda":
                self.pipeline.enable_model_cpu_offload() # Saves VRAM
                # self.pipeline.enable_xformers_memory_efficient_attention() # Optional if xformers installed

            logger.info("ControlNet Pipeline loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load ControlNet: {e}")
            raise e

    def process_image(self, image_bytes, prompt, negative_prompt="", controlnet_conditioning_scale=0.5, steps=20):
        if self.pipeline is None:
            self.load_model()

        try:
            # 1. Convert bytes to PIL Image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_image = Image.fromarray(image)

            # 2. Preprocess for Canny
            image = np.array(original_image)
            low_threshold = 100
            high_threshold = 200
            image = cv2.Canny(image, low_threshold, high_threshold)
            image = image[:, :, None]
            image = np.concatenate([image, image, image], axis=2)
            canny_image = Image.fromarray(image)

            # 3. Prompt Engineering & Inference
            # Inject quality boosters
            enhanced_prompt = prompt + ", highly detailed, 8k, masterpiece, photorealistic, trending on artstation, sharp focus"
            
            # Default negative prompt if none provided
            if not negative_prompt:
                negative_prompt = "low quality, bad quality, sketches, bad anatomy, deformed, disfigured, blurry, pixelated, extra limbs, missing limbs, bad hands, text, watermark, signature"

            output = self.pipeline(
                enhanced_prompt,
                image=canny_image,
                negative_prompt=negative_prompt,
                num_inference_steps=30 if steps < 30 else steps, # Ensure at least 30 steps for quality
                controlnet_conditioning_scale=controlnet_conditioning_scale,
            ).images[0]

            # 4. Return bytes
            img_byte_arr = io.BytesIO()
            output.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()

        except Exception as e:
            logger.error(f"ControlNet inference failed: {e}")
            raise e

# Singleton
controlnet_service = ControlNetService()
