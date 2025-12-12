import torch
import time
import sys

def verify_gpu_load():
    print("Checking CUDA availability...")
    if not torch.cuda.is_available():
        print("❌ CUDA is NOT available. Running on CPU.")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"✅ CUDA IS AVAILABLE! Device: {gpu_name}")
    print(f"Current Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

    print("\nStarting GPU Stress Test (10 seconds)...")
    print("WATCH YOUR TASK MANAGER NOW.")
    print("Look for 'Cuda' or 'Compute_0' graphs, or just a spike in overall usage.")
    
    # Create large matrices
    N = 10000
    a = torch.randn(N, N, device=device)
    b = torch.randn(N, N, device=device)

    start_time = time.time()
    iterations = 0
    
    try:
        while time.time() - start_time < 10:
            # Heavy matrix multiplication
            c = torch.matmul(a, b)
            torch.cuda.synchronize() # Wait for computation
            iterations += 1
            print(f"\rComputing iteration {iterations}...", end="")
    except KeyboardInterrupt:
        print("\nStopped.")

    print("\n\n✅ Test Complete.")
    print(f"Performed {iterations} massive matrix multiplications on GPU.")

if __name__ == "__main__":
    verify_gpu_load()
