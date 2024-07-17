from fastapi import APIRouter
from api.api_v1.endpoints import audio
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

api_router = APIRouter()

@api_router.get("/", response_class=JSONResponse)
async def ping() -> dict:
   
   return FileResponse('static/index.html')
    
    
api_router.include_router(audio.router, prefix="/audio", tags=["audio"])

