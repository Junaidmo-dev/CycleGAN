import requests
import os

def test_detect():
    url = "http://localhost:8001/api/v1/detect"
    # Use a dummy image or a real one if available. 
    # We'll create a dummy image using PIL
    from PIL import Image
    import io
    
    img = Image.new('RGB', (100, 100), color = 'red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    
    files = {'file': ('test.png', img_byte_arr, 'image/png')}
    
    try:
        response = requests.post(url, files=files)
        if response.status_code == 200:
            print("Success:", response.json())
        else:
            print(f"Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_detect()
