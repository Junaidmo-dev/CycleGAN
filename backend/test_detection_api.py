import requests

# Test the detection endpoint
url = "http://localhost:8001/api/v1/detect"

# Use a test image
with open("C:/Users/junai/.gemini/antigravity/brain/ec751dcd-0038-40b3-b975-18ae6fe97927/uploaded_image_1764423586258.png", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
