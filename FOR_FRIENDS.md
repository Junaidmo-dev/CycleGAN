# 🛠️ Setup Guide for Friends

Welcome! This guide will help you get the **DeepClean AI** project running on your local machine. Because this is a heavy AI project (models are ~10GB), certain large files are **not** included in the GitHub repository. Follow these steps carefully to get everything working.

---

## 📌 Prerequisites

Before starting, ensure you have the following installed:
*   [Python 3.10+](https://www.python.org/downloads/)
*   [Node.js 18+](https://nodejs.org/)
*   [Git](https://git-scm.com/)

---

## 🚀 Step 1: Clone & Navigate
If you haven't already, clone the repository and enter the folder:
```bash
git clone https://github.com/Junaidmo-dev/CycleGAN.git
cd CycleGAN
```

---

## 🐍 Step 2: Backend Setup (Python)

Open a terminal in the root folder and run:
```powershell
# 1. Enter backend folder
cd backend

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate it
# On Windows:
.\.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# 4. Install dependencies
# NOTE: To use your NVIDIA GPU (like MX350), use the CUDA command:
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## 🎨 Step 3: Frontend Setup (Next.js)

Open a **new** terminal in the root folder:
```bash
# 1. Enter frontend folder
cd frontend

# 2. Install dependencies
npm install
```

---

## 🧠 Step 4: Add the AI Models (CRITICAL)

The AI model weights are too large for GitHub. You need to put them in the `backend/models/` folder manually.

**You need these files:**
*   `generator.pth` (The main CycleGAN model)
*   `yolov8n_fish.pt` (Object detection model)
*   `raunenet_generator.pth` (Enhancement model)
*   `real_denoising.pth` (Denoising model)

> **Don't have the models?** 
> Ask the project owner (Junaid) for the `models` folder or run the helper script to create dummy models for testing:
> ```bash
> cd backend
> python setup_model.py
> ```
> *Choose Option 1 to generate test models.*

---

## ⚡ Step 5: Run the Project

You need to keep **two terminals** running:

### Terminal 1: Backend
```powershell
cd backend
.\.venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

### Terminal 2: Frontend
```powershell
cd frontend
npm run dev
```

---

## 🌐 Step 6: Open the App
Go to your browser and open:
👉 **[http://localhost:3000](http://localhost:3000)**

---

## 🛠️ Troubleshooting

*   **Port 8001 busy?** Change the port in the uvicorn command or kill the process using `netstat -ano | findstr :8001`.
*   **Module Not Found (e.g., 'realesrgan')?** This happens if some dependencies are missing. Ensure you have the latest code and run:
    ```bash
    pip install -r requirements.txt
    ```
*   **Hardware: NVIDIA GPU (MX350, RTX, etc.) not being used?**
    1.  Ensure you have [NVIDIA Drivers](https://www.nvidia.com/Download/index.aspx) installed.
    2.  Check if CUDA is detected by running:
        ```bash
        python verify_gpu.py
        ```
    3.  If it says "CUDA is NOT available", you likely installed the CPU-only version of PyTorch. Fix it by running:
        ```bash
        pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
        ```
    4.  **MX350 Note:** Since the MX350 usually has only 2GB of VRAM, if you encounter "Out of Memory" errors, the app is configured to use tiling to stay within limits.
*   **Virtual Environment activation:** Ensure your virtual environment is activated (`.venv\Scripts\activate`) before running the backend.
*   **Frontend Error?** Make sure the backend is running on port **8001**. The frontend is configured to talk to the backend on this specific port.

---
**Happy Coding! 🚀**
