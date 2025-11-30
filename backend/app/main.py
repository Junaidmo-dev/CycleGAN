from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .routes import enhance, health, jobs, detect, restormer
from .utils import logger
from .model_loader import model_loader

app = FastAPI(
    title=settings.APP_NAME,
    description="API for DeepClean AI Underwater Image Enhancer",
    version="1.0.0",
    debug=settings.DEBUG
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(enhance.router, prefix=settings.API_PREFIX, tags=["Enhancement"])
app.include_router(jobs.router, prefix=settings.API_PREFIX, tags=["Jobs"])
app.include_router(detect.router, prefix=settings.API_PREFIX, tags=["Detection"])
app.include_router(restormer.router, prefix=settings.API_PREFIX, tags=["Restormer"])
app.include_router(health.router, prefix=settings.API_PREFIX, tags=["Health"])

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up DeepClean AI Backend...")
    import sys
    logger.info(f"Python Executable: {sys.executable}")
    logger.info(f"System Path: {sys.path}")
    
    # Log all registered routes
    for route in app.routes:
        logger.info(f"Route: {route.path} [{route.name}]")

    model_loader.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")

@app.get("/")
async def root():
    return {"message": "Welcome to DeepClean AI API. Visit /docs for documentation."}

@app.get(f"{settings.API_PREFIX}/models")
async def get_models():
    """Return list of available models"""
    return {"models": model_loader.get_available_models()}

@app.get(f"{settings.API_PREFIX}/debug")
async def debug_import():
    """Debug endpoint to check imports"""
    import sys
    import os
    import traceback
    
    result = {
        "sys.executable": sys.executable,
        "cwd": os.getcwd(),
        "sys.path": sys.path,
        "import_status": "unknown",
        "error": None
    }
    
    try:
        # Re-attempt import logic
        img2img_base = os.path.join(os.path.dirname(__file__), "..", "img2img_turbo")
        src_path = os.path.join(img2img_base, "src")
        
        # Check if paths exist
        result["paths_exist"] = {
            "img2img_base": os.path.exists(img2img_base),
            "src_path": os.path.exists(src_path)
        }
        
        from src.pix2pix_turbo import Pix2Pix_Turbo
        result["import_status"] = "success"
        result["Pix2Pix_Turbo"] = str(Pix2Pix_Turbo)
    except Exception as e:
        result["import_status"] = "failed"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
        
    return result
