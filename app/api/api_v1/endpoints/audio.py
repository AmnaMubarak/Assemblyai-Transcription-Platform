from fastapi import APIRouter, FastAPI, UploadFile, File, Depends
from fastapi.responses import FileResponse, JSONResponse
import assemblyai as aai
from pydantic import BaseModel
from pathlib import Path
from schema.transcriptions import TranscriptionConfig
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("api_key")


app = FastAPI()  # Assuming you have an app instance
router = APIRouter()

# Replace with your AssemblyAI API key
aai.settings.api_key = api_key

class FileStorage:
    def __init__(self):
        self.file_path = None
        self.transcribed_text = None

# Create a single instance of FileStorage
file_storage = FileStorage()

# Dependency provider function
def get_file_storage():
    return file_storage

# Override the FileStorage dependency to always use the same instance
app.dependency_overrides[Depends(FileStorage)] = get_file_storage


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...), storage: FileStorage = Depends(get_file_storage)):
    global file_storage
    storage.file_path = f"uploaded_{file.filename}"
    with open(storage.file_path, "wb") as f:
        f.write(await file.read())
    return JSONResponse(content={"message": "File uploaded successfully"})

@router.post("/convert")
async def convert_audio(storage: FileStorage = Depends(get_file_storage)):
    if not storage.file_path or not Path(storage.file_path).is_file():
        return JSONResponse(content={"message": "No file uploaded"}, status_code=400)
    
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        sentiment_analysis=True,
        entity_detection=True,
        speaker_labels=True,
        speakers_expected=3,
        language_code="en_us"
    )
    config_dict = {
    "speech_model": config.speech_model,
    "sentiment_analysis": config.sentiment_analysis,
    "entity_detection": config.entity_detection,
    "speaker_labels": config.speaker_labels,
    "speakers_expected": config.speakers_expected,
    "language_code": config.language_code
}
    
    transcriber = aai.Transcriber(config=config_dict)
    # response = transcriber.transcribe(storage.file_path, **config_dict)
    transcriber = aai.Transcriber(config=config)
    response = transcriber.transcribe(storage.file_path)
    
    if response.status == 'error':
        return JSONResponse(content={"message": response.error}, status_code=500)
    else:
        storage.transcribed_text = "\n".join([f"Speaker {utt.speaker}: {utt.text}" for utt in response.utterances])
        return JSONResponse(content={"text": storage.transcribed_text})

@router.get("/download")
async def download_text(storage: FileStorage = Depends(get_file_storage)):
    if not storage.transcribed_text:
        return JSONResponse(content={"message": "No text available"}, status_code=400)
    
    text_file = "transcription.txt"
    with open(text_file, "w") as f:
        f.write(storage.transcribed_text)
    
    return FileResponse(text_file, media_type='application/octet-stream', filename=text_file)    

app.include_router(router)
