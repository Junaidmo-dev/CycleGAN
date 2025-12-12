
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

output_file = "source_dump.txt"

try:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("--- START INSPECTION ---\n")
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            revision=REVISION, 
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
            local_files_only=True
        )
        
        if hasattr(model, 'answer_question'):
            f.write("Source of answer_question:\n")
            f.write(inspect.getsource(model.answer_question))
            f.write("\n")
        else:
            f.write("model has no answer_question method\n")

        if hasattr(model, 'generate'):
            f.write("Source of generate:\n")
            f.write(inspect.getsource(model.generate))
            f.write("\n")
        else:
            f.write("model has no generate method\n")

        f.write("--- END INSPECTION ---\n")
        
    print(f"Dumped to {output_file}")

except Exception as e:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"ERROR: {e}\n")
    print(f"Error: {e}")
