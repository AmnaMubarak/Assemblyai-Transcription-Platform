from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from api.router import api_router
from dotenv import load_dotenv
import os
import logging
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app/logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("main_app")

# Load environment variables
load_dotenv()

# Get configuration from environment variables
DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "t")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",")

# Create FastAPI app with metadata for Swagger UI
app = FastAPI(
    title="Speech to Text Conversion API",
    description="An API for converting speech audio files to text using AssemblyAI",
    version="1.0",
    debug=DEBUG
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)    

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include API router
app.include_router(api_router)

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    # Run the application with uvicorn if this file is executed directly
    logger.info("Starting Speech-to-Text Conversion API")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)