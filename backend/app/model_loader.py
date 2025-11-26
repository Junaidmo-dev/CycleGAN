import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
from .config import settings
from .utils import logger


class ResidualBlock(nn.Module):
    """ResNet residual block with reflection padding"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """
    ResNet-based Generator for CycleGAN
    
    Architecture:
    - Encoder: 3 conv layers with downsampling
    - Transformer: 9 ResNet blocks
    - Decoder: 3 deconv layers with upsampling
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        """
        Args:
            input_nc: Number of input channels (3 for RGB)
            output_nc: Number of output channels (3 for RGB)
            ngf: Number of filters in the first conv layer
            n_blocks: Number of ResNet blocks (6 or 9)
        """
        super(ResnetGenerator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True)
            ]
        
        # ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResidualBlock(ngf * mult)]
        
        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                 kernel_size=3, stride=2, padding=1,
                                 output_padding=1, bias=False),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)


from .raunenet import RauneNet

# --- Global Model and Config Loading ---
MODEL_V = 'RAUNENet'
MODEL_NAME = 'test'
MODEL_EPOCH = 95
INPUT_WIDTH, INPUT_HEIGHT = 256, 256
NUM_DOWN, NUM_BLOCKS = 2, 30
CHECKPOINT_DIR = 'pretrained'

class ModelLoader:
    """Singleton class to load and cache the CycleGAN/RAUNENet model"""
    _instance = None
    model = None
    device = None
    model_type = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance.load_model()
        return cls._instance

    def load_model(self):
        """Load the model from disk or create a new one"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        try:
            # Determine which model to load
            # Check if RAUNENet weights exist
            raune_path = os.path.join(os.path.dirname(settings.MODEL_PATH), "raunenet_generator.pth")
            
            if MODEL_V == 'RAUNENet':
                logger.info(f"Initializing RAUNENet (blocks={NUM_BLOCKS}, down={NUM_DOWN})...")
                self.model = RauneNet(
                    input_nc=3,
                    output_nc=3,
                    n_blocks=NUM_BLOCKS,
                    n_down=NUM_DOWN,
                    use_att_up=False # Default from snippet/repo
                )
                self.model_type = "RauneNet"
                target_path = raune_path
            else:
                logger.info("Initializing Standard ResnetGenerator...")
                self.model = ResnetGenerator(
                    input_nc=3,
                    output_nc=3,
                    ngf=64,
                    n_blocks=9
                )
                self.model_type = "ResnetGenerator"
                target_path = settings.MODEL_PATH

            # Try to load pre-trained weights
            if os.path.exists(target_path):
                file_size = os.path.getsize(target_path)
                if file_size > 1_000_000:
                    try:
                        logger.info(f"Loading model from {target_path} ({file_size / 1_000_000:.2f} MB)")
                        state_dict = torch.load(target_path, map_location=self.device)
                        
                        # Handle state dict wrapping
                        if isinstance(state_dict, dict):
                            if 'model' in state_dict:
                                state_dict = state_dict['model']
                            elif 'state_dict' in state_dict:
                                state_dict = state_dict['state_dict']
                        
                        self.model.load_state_dict(state_dict)
                        logger.info(f"✅ {self.model_type} loaded successfully from disk!")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not load state dict: {e}")
                        logger.warning("Using randomly initialized model (untrained)")
                else:
                    logger.warning(f"⚠️ Model file too small. Using initialized model.")
            else:
                logger.warning(f"⚠️ Model file not found at {target_path}")
                if self.model_type == "RauneNet":
                    logger.warning("Please download RAUNENet weights and save as 'backend/models/raunenet_generator.pth'")
                
            self.model.to(self.device)
            self.model.eval()
            
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Model parameters: {total_params:,}")
            
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise e

    def get_model(self):
        return self.model, self.device, self.model_type


# Create singleton instance
model_loader = ModelLoader()
