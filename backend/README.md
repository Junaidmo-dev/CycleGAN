# DeepClean AI Backend

Production-grade backend for the DeepClean AI Underwater Image Enhancer.

## Features
- **FastAPI** for high-performance REST API
- **PyTorch** for CycleGAN and Restormer inference
- **Moondream 2** for automated marine life detection
- **Celery + Redis** for asynchronous background processing
- **Docker** support for easy deployment

## Setup

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run Redis (required for async tasks):
   ```bash
   docker run -d -p 6379:6379 redis
   ```

3. Start the worker:
   ```bash
   celery -A app.workers.celery_app worker --loglevel=info
   ```

4. Start the API:
   ```bash
   uvicorn app.main:app --reload
   ```

### Docker Deployment

1. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

## API Endpoints

- `POST /api/enhance`: Synchronous image enhancement
- `POST /api/enhance-async`: Asynchronous enhancement (returns job_id)
- `GET /api/jobs/{job_id}`: Check status of async job
- `GET /api/health`: Health check

## Configuration

Environment variables can be set in `.env` or passed to Docker.
See `app/config.py` for available settings.
