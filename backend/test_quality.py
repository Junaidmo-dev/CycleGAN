"""
Quick diagnostic to test img2img-turbo quality
"""
import sys
sys.path.insert(0, 'c:/Users/junai/Desktop/Major - Copy/backend')

from PIL import Image, ImageDraw
import io
from app.inference import run_img2img_turbo

# Create a simple test sketch
img = Image.new('RGB', (512, 512), color='white')
d = ImageDraw.Draw(img)
# Draw a fish shape
d.ellipse([150, 200, 350, 300], outline='black', width=5)  # Body
d.polygon([(350, 250), (400, 220), (400, 280)], outline='black', width=3)  # Tail
d.ellipse([180, 230, 200, 250], fill='black')  # Eye

# Save to bytes
buf = io.BytesIO()
img.save(buf, format='PNG')
img_bytes = buf.getvalue()

print("Running inference...")
print("Prompt: 'a blue fish swimming in the ocean with coral reef background'")

try:
    result_bytes = run_img2img_turbo(
        image_bytes=img_bytes,
        prompt="a blue fish swimming in the ocean with coral reef background",
        skip_canny=True
    )
    
    # Save result
    result_img = Image.open(io.BytesIO(result_bytes))
    result_img.save("test_quality_diagnostic.png")
    print(f"\n✅ SUCCESS! Result saved to: test_quality_diagnostic.png")
    print(f"Result size: {result_img.size}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
