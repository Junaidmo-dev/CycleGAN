import requests
import os
from PIL import Image, ImageDraw
import io

API_URL = "http://localhost:8001/api/v1/enhance"

def create_dummy_sketch():
    img = Image.new('RGB', (512, 512), color = 'white')
    d = ImageDraw.Draw(img)
    d.line([(100,100), (400,400)], fill="black", width=5)
    d.rectangle([(200, 200), (300, 300)], outline="black", width=5)
    
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def test_stochastic():
    print("Testing Stochastic Turbo...")
    img_bytes = create_dummy_sketch()
    
    # Test 1: Gamma 0.2
    print("Requesting Gamma 0.2...")
    files = {"file": ("test.png", img_bytes, "image/png")}
    data = {
        "model": "sketch_turbo_stochastic",
        "prompt": "a futuristic building",
        "gamma": 0.2,
        "skip_canny": True
    }
    
    try:
        response = requests.post(API_URL, files=files, data=data)
        if response.status_code == 200:
            print("✅ Gamma 0.2 Success")
            with open("test_stochastic_0.2.png", "wb") as f:
                f.write(response.content)
        else:
            print(f"❌ Gamma 0.2 Failed: {response.text}")
            return
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return

    # Test 2: Gamma 0.8
    print("Requesting Gamma 0.8...")
    files = {"file": ("test.png", img_bytes, "image/png")}
    data["gamma"] = 0.8
    
    try:
        response = requests.post(API_URL, files=files, data=data)
        if response.status_code == 200:
            print("✅ Gamma 0.8 Success")
            with open("test_stochastic_0.8.png", "wb") as f:
                f.write(response.content)
            print("Please visually compare test_stochastic_0.2.png and test_stochastic_0.8.png")
        else:
            print(f"❌ Gamma 0.8 Failed: {response.text}")

if __name__ == "__main__":
    test_stochastic()
