import torch
import torch.nn as nn
import os
import gc
from .config import settings
from .utils import logger
from .raunenet import RauneNet
from .cyclegan import ResnetGenerator
import sys



# --- Global Config ---
MODEL_EPOCH = 95
INPUT_WIDTH, INPUT_HEIGHT = 256, 256
NUM_DOWN, NUM_BLOCKS = 2, 30

class ModelLoader:
    """Singleton class to load and cache multiple models"""
    _instance = None
    models = {}
    current_model_name = None
    device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        """Initialize device"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
    def unload_all_models(self):
        """Unload all models from GPU to free memory"""
        if self.models:
            logger.info("🧹 Unloading all models to free VRAM...")
            self.models.clear()
            self.current_model_name = None
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info(f"   VRAM Free: {torch.cuda.memory_allocated() / 1024**3:.2f} GB used")

    def load_raunenet(self):
        """Load RAUNENet model"""
        try:
            logger.info(f"Initializing RAUNENet (blocks={NUM_BLOCKS}, down={NUM_DOWN})...")
            model = RauneNet(
                input_nc=3,
                output_nc=3,
                n_blocks=NUM_BLOCKS,
                n_down=NUM_DOWN,
                use_att_up=False
            )
            
            # Path to RAUNENet weights
            weights_path = os.path.join(os.path.dirname(settings.MODEL_PATH), "raunenet_generator.pth")
            
            if self._load_weights(model, weights_path):
                self.models['raunenet'] = model
                target_device = self.device
                model.to(target_device)
                self.current_model_name = 'raunenet'
                logger.info("✅ RAUNENet loaded successfully!")
            else:
                logger.warning("⚠️ RAUNENet weights not found or invalid. Model not loaded.")

        except Exception as e:
            logger.error(f"❌ Error loading RAUNENet: {e}")

    def load_cyclegan(self):
        """Load CycleGAN model"""
        try:
            logger.info("Initializing CycleGAN ResnetGenerator...")
            model = ResnetGenerator(
                input_nc=3,
                output_nc=3,
                ngf=64,
                n_blocks=9
            )
            
            # Path to CycleGAN weights
            weights_path = settings.MODEL_PATH 
            
            if self._load_weights(model, weights_path):
                self.models['cyclegan'] = model
                model.to(self.device)
                self.current_model_name = 'cyclegan'
                logger.info("✅ CycleGAN loaded successfully!")
            else:
                logger.warning("⚠️ CycleGAN weights not found or invalid. Model not loaded.")

        except Exception as e:
            logger.error(f"❌ Error loading CycleGAN: {e}")




    def _load_weights(self, model, path):
        """Helper to load weights into a model"""
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            if file_size > 1_000_000: # Basic check
                try:
                    logger.info(f"Loading weights from {path} ({file_size / 1_000_000:.2f} MB)")
                    state_dict = torch.load(path, map_location='cpu') # Load to CPU first
                    
                    # Handle state dict wrapping
                    if isinstance(state_dict, dict):
                        if 'model' in state_dict:
                            state_dict = state_dict['model']
                        elif 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                    
                    model.load_state_dict(state_dict, strict=False)
                    # Don't move to device here, let the loader do it
                    model.eval()
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ Could not load state dict from {path}: {e}")
                    return False
            else:
                logger.warning(f"⚠️ File too small: {path}")
                return False
        else:
            logger.warning(f"⚠️ File not found: {path}")
            return False

    def get_model(self, model_name='raunenet'):
        """Get a specific model by name, managing memory aggressively"""
        
        # Check if requested model is already loaded and active on GPU
        if model_name in self.models and self.current_model_name == model_name:
            return self.models[model_name], self.device, model_name
            
        # If different model is requested, UNLOAD EVERYTHING first
        # This is critical for 4GB VRAM
        if self.current_model_name != model_name:
            print(f"🔄 Switching models: {self.current_model_name} -> {model_name}")
            self.unload_all_models()
            
        # Load the requested model
        if model_name == "raunenet":
            self.load_raunenet()
        elif model_name == "cyclegan":
            self.load_cyclegan()
        else:
            raise ValueError(f"Unknown model requested: {model_name}")
            
        if model_name in self.models:
             return self.models[model_name], self.device, model_name
        else:
             raise RuntimeError(f"Failed to load model: {model_name}")

    def get_available_models(self):
        """Return list of available model names based on files"""
        # This is a bit of a lie now since we don't hold them in memory
        # But for the UI dropdown, we should return what CAN be loaded
        available = ['raunenet', 'cyclegan']
        
        return available


# Create singleton instance
model_loader = ModelLoader()
