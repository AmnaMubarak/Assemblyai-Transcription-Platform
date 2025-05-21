from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
import assemblyai as aai
from pathlib import Path
from schema.transcriptions import TranscriptionConfig
from dotenv import load_dotenv
import os
import logging
import shutil
import uuid
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app/logs/api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("audio_api")

# Load environment variables from .env file
load_dotenv()

# Access the API key securely
api_key = os.getenv("api_key")
if not api_key:
    logger.critical("AssemblyAI API key not found. Please add it to your .env file.")
    raise ValueError("AssemblyAI API key is required")

# Configure API key for AssemblyAI
aai.settings.api_key = api_key

# Get upload directory from environment or use default
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "app/uploads")
# Ensure the upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Set maximum file size (default: 10MB)
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", 10 * 1024 * 1024))  # 10MB in bytes

router = APIRouter()

class FileStorage:
    def __init__(self):
        self.file_path = None
        self.transcribed_text = None
        self.file_uuid = None

# Create a single instance of FileStorage
file_storage = FileStorage()

# Dependency provider function
def get_file_storage():
    return file_storage

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_audio(file: UploadFile = File(...), storage: FileStorage = Depends(get_file_storage)):
    """
    Upload an audio file for transcription.
    
    Args:
        file: The audio file to upload
        storage: The FileStorage dependency
        
    Returns:
        JSON response with a success message or error
    """
    try:
        # Validate file size
        file_size = 0
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size allowed is {MAX_UPLOAD_SIZE/1024/1024}MB"
            )
        
        # Validate file type
        content_type = file.content_type
        if not content_type or not content_type.startswith("audio/"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="File must be an audio file"
            )
            
        # Generate a unique ID for this file
        storage.file_uuid = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        storage.file_path = Path(UPLOAD_DIR) / f"{storage.file_uuid}{file_extension}"
        
        # Save the file
        with open(storage.file_path, "wb") as f:
            f.write(file_content)
            
        logger.info(f"File uploaded successfully: {storage.file_path}")
        return JSONResponse(content={"message": "File uploaded successfully", "file_id": storage.file_uuid})
        
    except HTTPException as e:
        # Re-raise HTTP exceptions to maintain their status codes
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while uploading the file: {str(e)}"
        )

@router.post("/convert", status_code=status.HTTP_200_OK)
async def convert_audio(storage: FileStorage = Depends(get_file_storage)):
    """
    Convert an uploaded audio file to text using AssemblyAI.
    
    Args:
        storage: The FileStorage dependency containing the file path
        
    Returns:
        JSON response with a success message or error
    """
    try:
        # Check if a file was uploaded
        if not storage.file_path or not Path(storage.file_path).is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No file uploaded or file not found. Please upload a file first."
            )
        
        # Configure and initialize the transcription
        logger.info(f"Starting transcription for file: {storage.file_path}")
        start_time = time.time()
        
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            sentiment_analysis=True,
            entity_detection=True,
            speaker_labels=True,
            speakers_expected=3,
            language_code="en_us"
        )
        
        transcriber = aai.Transcriber(config=config)
        response = transcriber.transcribe(str(storage.file_path))
        
        # Check for errors in the response
        if response.status == 'error':
            logger.error(f"Transcription error: {response.error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Transcription failed: {response.error}"
            )
        
        # Format the transcription with speaker labels
        storage.transcribed_text = "\n".join([f"Speaker {utt.speaker}: {utt.text}" for utt in response.utterances])
        
        # Log completion
        elapsed_time = time.time() - start_time
        logger.info(f"Transcription completed successfully in {elapsed_time:.2f} seconds")
        
        return JSONResponse(content={"message": "Transcription successful. Please download the transcript."})
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during transcription: {str(e)}"
        )

@router.get("/download", status_code=status.HTTP_200_OK)
async def download_text(storage: FileStorage = Depends(get_file_storage)):
    """
    Download the transcribed text as a file.
    
    Args:
        storage: The FileStorage dependency containing the transcribed text
        
    Returns:
        A file response with the transcription or an error
    """
    try:
        # Check if text is available
        if not storage.transcribed_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No transcription available. Please convert an audio file first."
            )
        
        # Create a unique filename for the transcription
        if storage.file_uuid:
            text_file = Path(UPLOAD_DIR) / f"transcription_{storage.file_uuid}.txt"
        else:
            text_file = Path(UPLOAD_DIR) / f"transcription_{str(uuid.uuid4())}.txt"
        
        # Write the transcription to a file
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(storage.transcribed_text)
        
        logger.info(f"Transcription saved to file: {text_file}")
        
        # Return the file for download
        return FileResponse(
            path=str(text_file),
            media_type='text/plain',
            filename=text_file.name
        )
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during text download: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while preparing the download: {str(e)}"
        )

# Add a cleanup endpoint to free resources
@router.delete("/cleanup", status_code=status.HTTP_200_OK)
async def cleanup_resources(storage: FileStorage = Depends(get_file_storage)):
    """
    Clean up resources associated with the current session.
    
    Args:
        storage: The FileStorage dependency to reset
        
    Returns:
        JSON confirmation of cleanup
    """
    try:
        # Remove the uploaded file if it exists
        if storage.file_path and Path(storage.file_path).is_file():
            Path(storage.file_path).unlink()
            logger.info(f"Deleted file: {storage.file_path}")
        
        # Reset the storage object
        storage.file_path = None
        storage.transcribed_text = None
        storage.file_uuid = None
        
        return JSONResponse(content={"message": "Resources cleaned up successfully"})
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during cleanup: {str(e)}"
        )
