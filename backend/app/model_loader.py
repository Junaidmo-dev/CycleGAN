import torch
import torch.nn as nn
import os
from .config import settings
from .utils import logger
from .raunenet import RauneNet
from .cyclegan import ResnetGenerator
import sys

# Add img2img_turbo to path
img2img_base = os.path.join(os.path.dirname(__file__), "..", "img2img_turbo")
sys.path.append(img2img_base)
# CRITICAL: Add src to path so internal imports in pix2pix_turbo.py work
# The original code uses `from model import ...` which requires src to be in path
sys.path.append(os.path.join(img2img_base, "src"))

try:
    from src.pix2pix_turbo import Pix2Pix_Turbo
except ImportError as e:
    Pix2Pix_Turbo = None
    # Log the full error to help debugging
    logger.warning(f"[WARNING] Could not import Pix2Pix_Turbo: {e}")
    import traceback
    logger.warning(traceback.format_exc())

# --- Global Config ---
MODEL_EPOCH = 95
INPUT_WIDTH, INPUT_HEIGHT = 256, 256
NUM_DOWN, NUM_BLOCKS = 2, 30

class ModelLoader:
    """Singleton class to load and cache multiple models"""
    _instance = None
    models = {}
    device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        """Initialize device and load available models"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self.load_raunenet()
        self.load_cyclegan()
        self.load_transfiguration()
        self.load_img2img_turbo()

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
                logger.info("✅ CycleGAN loaded successfully!")
            else:
                logger.warning("⚠️ CycleGAN weights not found or invalid. Model not loaded.")

        except Exception as e:
            logger.error(f"❌ Error loading CycleGAN: {e}")

    def load_transfiguration(self):
        """Load Transfiguration models (CycleGAN architecture) dynamically"""
        try:
            transfiguration_dir = os.path.join(os.path.dirname(settings.MODEL_PATH), "transfiguration")
            if not os.path.exists(transfiguration_dir):
                os.makedirs(transfiguration_dir, exist_ok=True)
                logger.info(f"Created transfiguration directory: {transfiguration_dir}")
                return

            logger.info(f"Scanning for Transfiguration models in {transfiguration_dir}...")
            
            for filename in os.listdir(transfiguration_dir):
                if filename.endswith(".pth"):
                    model_name = f"transfiguration_{filename[:-4]}" # e.g., transfiguration_horse2zebra
                    logger.info(f"Initializing Transfiguration Model: {model_name}...")
                    
                    model = ResnetGenerator(
                        input_nc=3,
                        output_nc=3,
                        ngf=64,
                        n_blocks=9
                    )
                    
                    weights_path = os.path.join(transfiguration_dir, filename)
                    
                    if self._load_weights(model, weights_path):
                        self.models[model_name] = model
                        logger.info(f"✅ {model_name} loaded successfully!")
                    else:
                        logger.warning(f"⚠️ Failed to load weights for {model_name}")

        except Exception as e:
            logger.error(f"❌ Error loading Transfiguration models: {e}")

    def load_img2img_turbo(self, model_name="edge_to_image"):
        """Register img2img-turbo (edge_to_image or sketch_to_image_stochastic)"""
        try:
            if Pix2Pix_Turbo is None:
                logger.warning("[WARNING] Pix2Pix_Turbo class not available. Check import errors above.")
                return

            logger.info("="*60)
            logger.info(f"Initializing Pix2Pix_Turbo ({model_name})...")
            logger.info("This will auto-download weights if not present in checkpoints/")
            
            # Initialize model (auto-downloads weights to checkpoints/)
            model = Pix2Pix_Turbo(model_name)
            
            # Use the custom set_eval() method (not PyTorch's standard eval())
            # This properly freezes gradients and sets all submodules to eval mode
            model.set_eval()
            
            # Log device info to verify GPU usage
            logger.info(f"✓ Model initialized on device: {model.device}")
            if hasattr(model, 'unet'):
                 logger.info(f"✓ UNet device: {next(model.unet.parameters()).device}")
            
            # Store with specific key
            key = "img2img_turbo" if model_name == "edge_to_image" else "sketch_turbo_stochastic"
            self.models[key] = model
            logger.info(f"[SUCCESS] {key} loaded successfully!")
            logger.info("="*60)
        except Exception as e:
            logger.error(f"[ERROR] Failed to load img2img-turbo: {e}")
            import traceback
            logger.error(traceback.format_exc())


    def _load_weights(self, model, path):
        """Helper to load weights into a model"""
        if os.path.exists(path):
            file_size = os.path.getsize(path)
            if file_size > 1_000_000: # Basic check
                try:
                    logger.info(f"Loading weights from {path} ({file_size / 1_000_000:.2f} MB)")
                    state_dict = torch.load(path, map_location=self.device)
                    
                    # Handle state dict wrapping
                    if isinstance(state_dict, dict):
                        if 'model' in state_dict:
                            state_dict = state_dict['model']
                        elif 'state_dict' in state_dict:
                            state_dict = state_dict['state_dict']
                    
                    model.load_state_dict(state_dict, strict=False)
                    model.to(self.device)
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

    def load_controlnet(self):
        """Load ControlNet pipeline"""
        try:
            from .controlnet import controlnet_service
            controlnet_service.load_model()
            self.models['controlnet'] = controlnet_service
            logger.info("✅ ControlNet loaded successfully!")
        except Exception as e:
            logger.error(f"❌ Error loading ControlNet: {e}")

    def get_model(self, model_name='raunenet'):
        """Get a specific model by name"""
        
        if model_name not in self.models:
            # Lazy loading for turbo models
            if model_name == "img2img_turbo":
                self.load_img2img_turbo("edge_to_image")
            elif model_name == "sketch_turbo_stochastic":
                self.load_img2img_turbo("sketch_to_image_stochastic")
            elif model_name == "controlnet":
                self.load_controlnet()
            
            # Fallback logic for other models
            if model_name not in self.models:
                if self.models:
                    # If requesting a specific transfiguration that doesn't exist, try to find *any* transfiguration
                    if model_name.startswith("transfiguration_"):
                         logger.warning(f"Specific model '{model_name}' not found.")
                    
                    fallback = list(self.models.keys())[0]
                    logger.warning(f"Model '{model_name}' not found. Falling back to '{fallback}'")
                    return self.models[fallback], self.device, fallback
                else:
                    raise ValueError(f"No models loaded! Requested: {model_name}")
        
        return self.models[model_name], self.device, model_name

    def get_available_models(self):
        """Return list of available model names"""
        return list(self.models.keys())


# Create singleton instance
model_loader = ModelLoader()
