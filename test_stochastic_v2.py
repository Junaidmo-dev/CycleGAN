import requests
import io
from PIL import Image, ImageDraw

API_URL = "http://localhost:8001/api/v1/enhance"

def create_dummy_sketch():
    img = Image.new('RGB', (512, 512), color = 'white')
    d = ImageDraw.Draw(img)
    d.line([(100,100), (400,400)], fill="black", width=5)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def run():
    print("Testing Stochastic Turbo...")
    img_bytes = create_dummy_sketch()
    
    files = {"file": ("test.png", img_bytes, "image/png")}
    data = {"model": "sketch_turbo_stochastic", "prompt": "test", "gamma": 0.5, "skip_canny": True}
    
    try:
        res = requests.post(API_URL, files=files, data=data)
        print(f"Status: {res.status_code}")
        if res.status_code == 200:
            print("Success!")
        else:
            print(f"Failed: {res.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run()
