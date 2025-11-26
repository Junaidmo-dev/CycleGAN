import httpx
import sys

print("Testing backend API...")

try:
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    response = httpx.get("http://localhost:8000/api/health", timeout=5)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    # Test root endpoint
    print("\n2. Testing root endpoint...")
    response = httpx.get("http://localhost:8000/", timeout=5)
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.json()}")
    
    print("\n✅ Backend is responding!")
    
except httpx.TimeoutException:
    print("\n❌ ERROR: Request timed out. Backend is not responding.")
    sys.exit(1)
except httpx.ConnectError:
    print("\n❌ ERROR: Could not connect to backend. Is it running?")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    sys.exit(1)
