
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-08-26"

print(f"Loading {MODEL_ID} revision={REVISION}...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        revision=REVISION, 
        trust_remote_code=True,
        torch_dtype=torch.float32, # Force float32 for CPU/Safety
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    
    print(f"Model class: {type(model)}")
    print(f"Has generate? {hasattr(model, 'generate')}")
    print(f"Has encode_image? {hasattr(model, 'encode_image')}")
    print(f"Has answer_question? {hasattr(model, 'answer_question')}")
    
    # Try a dummy generation
    print("Attempting dummy generation...")
    # Moondream expects an image embedding for answer_question, so we can't easily call it without an image.
    # But checking attributes is a good start.

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
