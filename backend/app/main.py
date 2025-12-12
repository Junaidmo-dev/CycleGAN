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


