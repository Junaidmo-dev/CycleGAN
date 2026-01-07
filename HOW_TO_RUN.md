# 🚀 How to Run DeepClean AI (Without Docker)

## ✅ Quick Start Guide

### Prerequisites
- Python 3.8+ installed
- Node.js 18+ installed
- Redis (optional - only needed for async tasks)

---

## 🔧 Backend Setup

### Terminal 1: Backend API Server

```powershell
⚠️ **Note:** The frontend uses the synchronous `/api/enhance` endpoint, so Celery is **NOT required** for basic functionality.

If you want to use async enhancement (`/api/enhance-async`):

```powershell
# Navigate to backend directory
cd "c:\Users\junai\Desktop\Major - Copy\backend"

# Activate virtual environment
.venv\Scripts\activate

# Start Celery worker with Windows-compatible pool
celery -A app.workers.celery_app worker --loglevel=info --pool=solo
```

**Important:** Always use `--pool=solo` on Windows to avoid permission errors.

---

## 🎨 Frontend Setup


---

## 🐛 Troubleshooting

### Frontend stuck on "Enhancing your image..."

**Cause:** Backend API is not responding

**Solution:**
1. Check if backend is running on port 8001
2. Restart the uvicorn server:
   ```powershell
   # Kill existing processes
   Stop-Process -Name "uvicorn" -Force
   
   # Restart the server
   cd "c:\Users\junai\Desktop\Major - Copy\backend"
   .venv\Scripts\activate
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
   ```

### Celery Permission Errors on Windows

**Error:** `PermissionError: [WinError 5] Access is denied`

**Solution:** Always use `--pool=solo` flag:
```powershell
celery -A app.workers.celery_app worker --loglevel=info --pool=solo
```

### Model Warning: "Could not load state dict"

**Warning:** `WARNING - Could not load state dict: invalid load key, '#'.. Using initialized dummy model.`

**Explanation:** This means the CycleGAN model file is missing or invalid. The app will still run but use a dummy model that won't perform actual enhancement.

**Solution:** 
- Train a CycleGAN model and save it to `backend/models/generator.pth`, or
- Download a pre-trained model and place it in that location

### Port Already in Use

**Error:** `Address already in use`

**Solution:**
```powershell
# Find process using port 8001
netstat -ano | findstr :8001

# Kill the process (replace PID with actual process ID)
Stop-Process -Id <PID> -Force
```

---

## 📋 Quick Test

Test if backend is responding:

```powershell
cd "c:\Users\junai\Desktop\Major - Copy\backend"
python test_api.py
```

Expected output:
```
✅ Backend is responding!
```

---

## 🔄 Restart Everything

If things get messy, here's how to restart everything cleanly:

```powershell
# 1. Stop all processes
Stop-Process -Name "uvicorn" -Force -ErrorAction SilentlyContinue
Stop-Process -Name "node" -Force -ErrorAction SilentlyContinue

# 2. Start backend
cd "c:\Users\junai\Desktop\Major - Copy\backend"
.venv\Scripts\activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001

# 3. In a new terminal, start frontend
cd "c:\Users\junai\Desktop\Major - Copy\frontend"
npm run dev
```

---

## 📝 Notes

- **Celery is optional** - The frontend uses synchronous enhancement by default
- **Redis is optional** - Only needed if you want to use async tasks
- **Model file** - The app works without a real model (uses dummy model for testing)
- **CORS** - Already configured to allow requests from localhost:3000

---

## ✨ Features

- ✅ Synchronous image enhancement (no Celery needed)
- ✅ Asynchronous enhancement with job tracking (requires Celery + Redis)
- ✅ Auto-generated API documentation at `/docs`
- ✅ Health check endpoint at `/api/health`
- ✅ Beautiful Next.js frontend with drag-and-drop upload
- ✅ Before/after comparison view
- ✅ Download enhanced images

---

**Built with ❤️ using FastAPI, PyTorch, and Next.js**
