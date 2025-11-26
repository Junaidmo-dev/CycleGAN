import httpx
import sys
from PIL import Image
import io

# Create a dummy image
img = Image.new('RGB', (100, 100), color = 'red')
img_byte_arr = io.BytesIO()
img.save(img_byte_arr, format='PNG')
img_byte_arr.seek(0)

print("🚀 Sending test image to backend...")

try:
    files = {'file': ('test.png', img_byte_arr, 'image/png')}
    response = httpx.post("http://localhost:8001/api/enhance", files=files, timeout=30)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Success! Image enhanced.")
        print("\n📊 Execution Metadata (from headers):")
        print(f"   Model Type: {response.headers.get('X-Model-Type')}")
        print(f"   Device: {response.headers.get('X-Device')}")
        print(f"   Input Shape: {response.headers.get('X-Input-Shape')}")
        print(f"   Status: {response.headers.get('X-Execution-Status')}")
        
        print("\n🔍 All Headers:")
        for k, v in response.headers.items():
            print(f"   {k}: {v}")
    else:
        print(f"❌ Failed: {response.text}")
        
except Exception as e:
    print(f"❌ Error: {e}")
