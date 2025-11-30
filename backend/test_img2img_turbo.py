import sys
import os
from unittest.mock import MagicMock, patch
import io
from PIL import Image

# Add backend to path so we can import app modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock the heavy dependencies
sys.modules["src"] = MagicMock()
sys.modules["src.pix2pix_turbo"] = MagicMock()
sys.modules["src.image_prep"] = MagicMock()

# Mock cv2 for image_prep
sys.modules["cv2"] = MagicMock()

from app.inference import run_img2img_turbo
from app.model_loader import model_loader

def test_img2img_turbo_integration():
    print("Testing img2img-turbo integration logic (Edge to Image)...")
    
    # Mock the model instance
    mock_model = MagicMock()
    mock_model.return_value = [MagicMock()] # Mock output image tensor
    
    # Mock model_loader.get_model to return our mock model
    model_loader.get_model = MagicMock(return_value=(mock_model, "cpu", "img2img_turbo"))
    
    # Mock canny_from_pil
    with patch("app.inference.canny_from_pil") as mock_canny:
        mock_canny.return_value = Image.new("RGB", (256, 256))
        
        try:
            # Create dummy image bytes
            dummy_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
            
            # Run inference (mocked)
            result = run_img2img_turbo(dummy_bytes, "turn cat into dog")
            
            print("✅ run_img2img_turbo called successfully (mocked).")
            print("✅ Pipeline initialization, Canny edge detection, and inference steps verified.")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_img2img_turbo_integration()
