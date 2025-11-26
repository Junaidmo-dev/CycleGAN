"""
Script to create or download a CycleGAN model for DeepClean AI

This script provides options to:
1. Create a properly initialized (untrained) model for testing
2. Download a pre-trained model from a URL
3. Validate an existing model file
"""

import torch
import sys
import os
from pathlib import Path

# Add parent directory to path to import model_loader
sys.path.insert(0, str(Path(__file__).parent))

from app.model_loader import ResnetGenerator


def create_initialized_model(output_path: str):
    """
    Create a properly initialized CycleGAN model (untrained)
    
    This is useful for testing the application flow without a trained model.
    The model will process images but won't perform meaningful enhancement.
    """
    print("🎨 Creating initialized CycleGAN model...")
    
    # Create the model
    model = ResnetGenerator(
        input_nc=3,
        output_nc=3,
        ngf=64,
        n_blocks=9
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    # Save the model
    torch.save(model.state_dict(), output_path)
    
    # Check file size
    file_size = os.path.getsize(output_path)
    print(f"   File size: {file_size / 1_000_000:.2f} MB")
    print(f"✅ Model saved to: {output_path}")
    print("\n⚠️  Note: This is an UNTRAINED model. It will process images but won't")
    print("   perform meaningful enhancement. For real results, you need a trained model.")


def validate_model(model_path: str):
    """Validate that a model file can be loaded"""
    print(f"🔍 Validating model at: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: File not found at {model_path}")
        return False
    
    file_size = os.path.getsize(model_path)
    print(f"   File size: {file_size / 1_000_000:.2f} MB")
    
    if file_size < 1_000_000:
        print(f"⚠️  Warning: File is very small ({file_size} bytes)")
        print("   Expected size: > 40 MB for a real CycleGAN model")
        return False
    
    try:
        # Try to load the state dict
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Handle different save formats
        if isinstance(state_dict, dict):
            if 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
        
        # Create model and try to load
        model = ResnetGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=9)
        model.load_state_dict(state_dict)
        
        print("✅ Model loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nPossible issues:")
        print("  - Model architecture doesn't match")
        print("  - File is corrupted")
        print("  - Wrong file format")
        return False


def download_model(url: str, output_path: str):
    """Download a model from a URL"""
    print(f"📥 Downloading model from: {url}")
    
    try:
        import requests
        from tqdm import tqdm
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        
        print(f"✅ Downloaded to: {output_path}")
        
        # Validate the downloaded model
        return validate_model(output_path)
        
    except ImportError:
        print("❌ Error: 'requests' and 'tqdm' packages required for downloading")
        print("   Install with: pip install requests tqdm")
        return False
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False


def main():
    """Main function with interactive menu"""
    model_path = Path(__file__).parent / "models" / "generator.pth"
    
    print("=" * 60)
    print("🎨 DeepClean AI - CycleGAN Model Setup")
    print("=" * 60)
    print()
    print("Choose an option:")
    print()
    print("1. Create initialized model (for testing)")
    print("2. Download pre-trained model from URL")
    print("3. Validate existing model")
    print("4. Exit")
    print()
    
    choice = input("Enter your choice (1-4): ").strip()
    
    if choice == "1":
        print()
        create_initialized_model(str(model_path))
        print()
        print("🚀 Next steps:")
        print("   1. Restart the backend server")
        print("   2. Upload an image to test the flow")
        print("   3. Note: Results won't be meaningful (untrained model)")
        
    elif choice == "2":
        print()
        url = input("Enter model URL: ").strip()
        if url:
            download_model(url, str(model_path))
        else:
            print("❌ No URL provided")
    
    elif choice == "3":
        print()
        validate_model(str(model_path))
    
    elif choice == "4":
        print("Goodbye!")
        return
    
    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    main()
