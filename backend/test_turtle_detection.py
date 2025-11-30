import requests
from PIL import Image

# Test the detection endpoint with the turtle image
url = "http://localhost:8001/api/v1/detect"

image_path = "C:/Users/junai/.gemini/antigravity/brain/ec751dcd-0038-40b3-b975-18ae6fe97927/uploaded_image_1764423802807.png"

# Check image first
img = Image.open(image_path)
print(f"Image size: {img.size}")
print(f"Image mode: {img.mode}")

# Test API
with open(image_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    
print(f"\nStatus Code: {response.status_code}")
print(f"Response: {response.json()}")

# Check if detections is empty
data = response.json()
if len(data.get('detections', [])) == 0:
    print("\n⚠️ No detections found!")
else:
    print(f"\n✅ Found {len(data['detections'])} detections")
    for i, det in enumerate(data['detections']):
        print(f"  {i+1}. {det['label']} - {det['confidence']:.2%}")
