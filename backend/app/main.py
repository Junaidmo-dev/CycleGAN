from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .routes import enhance, health, jobs
from .utils import logger

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
app.include_router(health.router, prefix=settings.API_PREFIX, tags=["Health"])

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up DeepClean AI Backend...")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")

@app.get("/")
async def root():
    return {"message": "Welcome to DeepClean AI API. Visit /docs for documentation."}
