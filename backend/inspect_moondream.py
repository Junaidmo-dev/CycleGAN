
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import torch
from transformers import AutoModelForCausalLM
import inspect
import sys

MODEL_ID = "vikhyatk/moondream2"
REVISION = "2024-08-26"

print("--- START INSPECTION ---")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        revision=REVISION, 
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="cpu",
        local_files_only=True # Try to avoid redownload messages
    )
    
    if hasattr(model, 'answer_question'):
        print("Source of answer_question:")
        print(inspect.getsource(model.answer_question))
    else:
        print("model has no answer_question method")

    # Also check if it has a custom generate method that got hidden/lost?
    if hasattr(model, 'generate'):
        print("Source of generate:")
        print(inspect.getsource(model.generate))
    else:
        print("model has no generate method")

except Exception as e:
    print(f"ERROR: {e}")

print("--- END INSPECTION ---")
