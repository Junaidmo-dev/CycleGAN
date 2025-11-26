# 🎨 Adding a CycleGAN Model to DeepClean AI

This guide explains how to add a CycleGAN model for underwater image enhancement.

---

## 📋 **Prerequisites**

Your model file should be:
- A PyTorch `.pth` or `.pt` file
- Containing a CycleGAN Generator network
- Trained on underwater image enhancement (or similar domain transfer task)

---

## 🚀 **Option 1: Download a Pre-trained Model**

### **From Research Papers/GitHub**

Many researchers share pre-trained CycleGAN models. Here are some sources:

1. **Official CycleGAN Repository**
   - URL: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
   - They provide pre-trained models for various tasks
   - Download and adapt to your needs

2. **Underwater Image Enhancement Models**
   - Search GitHub for: "underwater image enhancement cyclegan pytorch"
   - Look for repositories with pre-trained weights
   - Common datasets: EUVP, UFO-120, UIEB

3. **Hugging Face Model Hub**
   - URL: https://huggingface.co/models
   - Search for "cyclegan" or "underwater enhancement"
   - Download `.pth` or `.pt` files

### **Steps to Add Downloaded Model:**

```bash
# 1. Download the model file (e.g., generator_underwater.pth)

# 2. Place it in the models directory
# Copy to: c:\Users\junai\Desktop\Major - Copy\backend\models\generator.pth

# 3. Restart the backend server
```

---

## 🏋️ **Option 2: Train Your Own Model**

### **Why Train Your Own?**
- Best results for your specific use case
- Full control over the model architecture
- Can use custom datasets

### **Training Steps:**

#### **1. Set Up Training Environment**

```bash
# Clone the official CycleGAN repository
git clone https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix

# Install dependencies
pip install -r requirements.txt
```

#### **2. Prepare Your Dataset**

Your dataset should have two domains:
- **Domain A**: Underwater images (degraded)
- **Domain B**: Clear/enhanced images (or above-water images)

Directory structure:
```
datasets/underwater/
├── trainA/          # Underwater images
├── trainB/          # Clear images
├── testA/           # Test underwater images
└── testB/           # Test clear images
```

Popular underwater datasets:
- **EUVP Dataset**: https://irvlab.cs.umn.edu/resources/euvp-dataset
- **UFO-120**: Underwater image dataset
- **UIEB**: Underwater Image Enhancement Benchmark

#### **3. Train the Model**

```bash
# Train CycleGAN
python train.py \
  --dataroot ./datasets/underwater \
  --name underwater_cyclegan \
  --model cycle_gan \
  --batch_size 1 \
  --gpu_ids 0 \
  --display_id -1

# Training will take several hours to days depending on:
# - Dataset size
# - GPU power
# - Number of epochs (default: 200)
```

#### **4. Export the Trained Model**

After training, the model will be saved in:
```
checkpoints/underwater_cyclegan/latest_net_G_A.pth
```

Copy this file to your DeepClean AI backend:
```bash
cp checkpoints/underwater_cyclegan/latest_net_G_A.pth \
   c:/Users/junai/Desktop/Major\ -\ Copy/backend/models/generator.pth
```

---

## 🔧 **Option 3: Use a Simple Test Model**

For testing purposes, I can help you create a proper CycleGAN architecture that will at least load correctly (even if not trained).

This is useful for:
- Testing the application flow
- Debugging
- Demonstrating the UI

The model won't perform real enhancement but will process images correctly.

---

## 📦 **Model Architecture Requirements**

Your model file should contain a PyTorch `state_dict` with the following structure:

### **For ResNet-based Generator (most common):**

```python
import torch
import torch.nn as nn

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super(ResnetGenerator, self).__init__()
        # ... (full architecture)
        
    def forward(self, x):
        return self.model(x)
```

### **Key Parameters:**
- `input_nc`: Number of input channels (3 for RGB)
- `output_nc`: Number of output channels (3 for RGB)
- `ngf`: Number of generator filters (typically 64)
- `n_blocks`: Number of ResNet blocks (6 or 9)

---

## 🔄 **Updating the Model Loader**

Your current `model_loader.py` uses a dummy model. You need to update it to use a proper CycleGAN architecture.

I'll create an updated version that:
1. Defines a proper ResNet-based Generator
2. Loads the state dict correctly
3. Handles missing/invalid models gracefully

---

## 🧪 **Testing Your Model**

After adding the model:

```bash
# 1. Restart the backend
cd "c:\Users\junai\Desktop\Major - Copy\backend"
.venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 2. Check the logs for:
# ✅ "Model loaded successfully from disk."
# ❌ "Could not load state dict..." (means model is incompatible)

# 3. Test via the API
python test_api.py

# 4. Test via the frontend
# Open http://localhost:3000 and upload an image
```

---

## 📊 **Expected Model File Size**

- **Small model (6 ResNet blocks)**: ~40-50 MB
- **Standard model (9 ResNet blocks)**: ~50-70 MB
- **Large model**: 100+ MB

Your current dummy file is only 20 bytes, so you'll know when you have a real model!

---

## 🐛 **Troubleshooting**

### **Error: "Could not load state dict: invalid load key"**

**Cause:** The model file format doesn't match the expected architecture

**Solutions:**
1. Check if the model was saved with `torch.save(model.state_dict(), 'file.pth')`
2. Verify the architecture matches (ResNet vs UNet vs other)
3. Try loading with `torch.load()` to inspect the structure

### **Error: "size mismatch for..."**

**Cause:** The model architecture doesn't match the saved weights

**Solutions:**
1. Check the model parameters (ngf, n_blocks, etc.)
2. Ensure input/output channels match (3 for RGB)
3. Update the architecture definition to match the trained model

### **Model loads but produces bad results**

**Causes:**
- Model not trained on underwater images
- Wrong normalization
- Incompatible preprocessing

**Solutions:**
1. Verify the model was trained for underwater enhancement
2. Check the normalization values match training
3. Ensure image preprocessing is correct

---

## 🎯 **Quick Start: Get a Working Model Now**

If you want to test the app immediately:

1. **Download a pre-trained CycleGAN model** from:
   - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md#pretrained-models

2. **Or use a research model**:
   - Search "underwater image enhancement github pytorch model"
   - Look for `.pth` files in the releases or checkpoints

3. **Place it in the models folder**:
   ```bash
   # Replace the dummy file
   # Path: c:\Users\junai\Desktop\Major - Copy\backend\models\generator.pth
   ```

4. **Restart the backend** and test!

---

## 📝 **Next Steps**

Would you like me to:

1. ✅ **Update the model_loader.py** with a proper CycleGAN architecture?
2. ✅ **Create a script to download a pre-trained model** automatically?
3. ✅ **Generate a properly initialized model** for testing (untrained but functional)?
4. ✅ **Help you set up training** with the official CycleGAN repository?

Let me know which option you'd prefer!

---

**Built with ❤️ for underwater image enhancement**
