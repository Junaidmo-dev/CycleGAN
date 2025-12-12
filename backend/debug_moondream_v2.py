
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-08-26"

print("--- START DEBUG ---")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        revision=REVISION, 
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    print(f"Model type: {type(model)}")
    print(f"Model attributes: {dir(model)}")
    
    gen_exists = hasattr(model, 'generate')
    print(f"HAS_GENERATE: {gen_exists}")
    
    if not gen_exists:
        print("CRITICAL: generate method missing!")
    
except Exception as e:
    print(f"ERROR: {e}")

print("--- END DEBUG ---")
