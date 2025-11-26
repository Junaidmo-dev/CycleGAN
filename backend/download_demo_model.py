import httpx
import shutil
from pathlib import Path
import sys

# URL for the official CycleGAN pre-trained model (Monet style)
MODEL_URL = "http://efrosgans.eecs.berkeley.edu/cyclegan/pretrained_models/style_monet.pth"
MODEL_PATH = Path("models/generator.pth")

def download_file(url, destination):
    print(f"⬇️ Downloading from {url}...")
    try:
        with httpx.stream("GET", url) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f:
                downloaded = 0
                for chunk in r.iter_bytes():
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percent = (downloaded / total_size) * 100
                        print(f"   Progress: {percent:.1f}%", end='\r')
                        
        print("\n✅ Download complete!")
        return True
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        return False

if __name__ == "__main__":
    # Ensure models directory exists
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if download_file(MODEL_URL, MODEL_PATH):
        print(f"🎉 Model saved to {MODEL_PATH.absolute()}")
        print("🚀 Restart the backend to load the new model!")
    else:
        sys.exit(1)
