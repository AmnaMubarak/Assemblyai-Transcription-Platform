from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, Form
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
import json
from typing import List, Dict, Optional
import zipfile
import io

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
# Set maximum batch size (default: 5 files)
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", 5))

router = APIRouter()

class FileStorage:
    def __init__(self):
        self.file_path = None
        self.transcribed_text = None
        self.file_uuid = None
        # New properties for batch processing
        self.batch_files: Dict[str, Dict] = {}
        self.batch_id: Optional[str] = None

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

@router.post("/batch-upload", status_code=status.HTTP_201_CREATED)
async def upload_batch(files: List[UploadFile] = File(...), storage: FileStorage = Depends(get_file_storage)):
    """
    Upload multiple audio files for batch processing.
    
    Args:
        files: List of audio files to upload
        storage: The FileStorage dependency
        
    Returns:
        JSON response with success message and batch ID
    """
    try:
        # Check batch size limit
        if len(files) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Too many files. Maximum allowed is {MAX_BATCH_SIZE} files per batch."
            )
        
        # Create a new batch ID
        batch_id = str(uuid.uuid4())
        storage.batch_id = batch_id
        storage.batch_files = {}
        
        # Process each file in the batch
        for file in files:
            # Validate file size
            file_content = await file.read()
            file_size = len(file_content)
            
            if file_size > MAX_UPLOAD_SIZE:
                # Skip oversized files but continue processing
                logger.warning(f"File {file.filename} exceeds size limit and will be skipped")
                continue
            
            # Validate file type
            content_type = file.content_type
            if not content_type or not content_type.startswith("audio/"):
                # Skip invalid files but continue processing
                logger.warning(f"File {file.filename} is not an audio file and will be skipped")
                continue
                
            # Generate a unique ID for this file
            file_uuid = str(uuid.uuid4())
            file_extension = Path(file.filename).suffix
            file_path = Path(UPLOAD_DIR) / f"{file_uuid}{file_extension}"
            
            # Save the file
            with open(file_path, "wb") as f:
                f.write(file_content)
                
            # Store file info in batch
            storage.batch_files[file_uuid] = {
                "original_name": file.filename,
                "path": str(file_path),
                "status": "uploaded",
                "transcribed_text": None
            }
                
            logger.info(f"Batch file uploaded: {file.filename} -> {file_path}")
        
        # Check if any files were successfully uploaded
        if not storage.batch_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid files were found in the batch"
            )
            
        return JSONResponse(content={
            "message": f"Batch uploaded successfully with {len(storage.batch_files)} files",
            "batch_id": batch_id,
            "files_count": len(storage.batch_files)
        })
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error uploading batch: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while uploading the batch: {str(e)}"
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

@router.post("/batch-convert", status_code=status.HTTP_202_ACCEPTED)
async def convert_batch(storage: FileStorage = Depends(get_file_storage)):
    """
    Convert all files in a batch to text.
    
    Args:
        storage: The FileStorage dependency with batch files
        
    Returns:
        JSON response with batch processing status
    """
    try:
        # Check if batch exists
        if not storage.batch_id or not storage.batch_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No batch found. Please upload files first."
            )
        
        # Configure transcription settings
        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            sentiment_analysis=True,
            entity_detection=True,
            speaker_labels=True,
            speakers_expected=3,
            language_code="en_us"
        )
        
        transcriber = aai.Transcriber(config=config)
        
        # Start processing each file
        total_files = len(storage.batch_files)
        processed_files = 0
        failed_files = 0
        
        logger.info(f"Starting batch transcription for {total_files} files")
        batch_start_time = time.time()
        
        for file_id, file_info in storage.batch_files.items():
            try:
                file_path = file_info["path"]
                
                # Skip already processed files
                if file_info["status"] == "transcribed":
                    processed_files += 1
                    continue
                
                # Check if the file exists
                if not Path(file_path).is_file():
                    logger.warning(f"File not found: {file_path}")
                    file_info["status"] = "error"
                    file_info["error"] = "File not found"
                    failed_files += 1
                    continue
                
                # Start transcription
                logger.info(f"Transcribing file: {file_path}")
                file_start_time = time.time()
                
                # Transcribe the file
                response = transcriber.transcribe(file_path)
                
                # Check for errors
                if response.status == 'error':
                    logger.error(f"Transcription error for file {file_id}: {response.error}")
                    file_info["status"] = "error"
                    file_info["error"] = f"Transcription failed: {response.error}"
                    failed_files += 1
                    continue
                
                # Format and save the transcription
                transcribed_text = "\n".join([f"Speaker {utt.speaker}: {utt.text}" for utt in response.utterances])
                file_info["transcribed_text"] = transcribed_text
                file_info["status"] = "transcribed"
                
                # Save to file
                output_path = Path(UPLOAD_DIR) / f"transcription_{file_id}.txt"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(transcribed_text)
                
                file_info["output_path"] = str(output_path)
                
                # Log completion
                file_elapsed_time = time.time() - file_start_time
                logger.info(f"File {file_id} transcribed in {file_elapsed_time:.2f} seconds")
                
                processed_files += 1
                
            except Exception as e:
                logger.error(f"Error processing file {file_id}: {str(e)}")
                file_info["status"] = "error"
                file_info["error"] = str(e)
                failed_files += 1
        
        # Log batch completion
        batch_elapsed_time = time.time() - batch_start_time
        logger.info(f"Batch transcription completed in {batch_elapsed_time:.2f} seconds. "
                   f"Processed: {processed_files}, Failed: {failed_files}")
        
        return JSONResponse(content={
            "message": "Batch processing completed",
            "batch_id": storage.batch_id,
            "total_files": total_files,
            "processed_files": processed_files,
            "failed_files": failed_files
        })
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during batch processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during batch processing: {str(e)}"
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

@router.get("/batch-download", status_code=status.HTTP_200_OK)
async def download_batch(storage: FileStorage = Depends(get_file_storage)):
    """
    Download all transcribed files in a batch as a ZIP archive.
    
    Args:
        storage: The FileStorage dependency with batch files
        
    Returns:
        A ZIP file with all transcriptions
    """
    try:
        # Check if batch exists and has transcribed files
        if not storage.batch_id or not storage.batch_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No batch found. Please upload and convert files first."
            )
        
        # Count transcribed files
        transcribed_files = [f for f in storage.batch_files.values() if f["status"] == "transcribed"]
        if not transcribed_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No transcribed files available in this batch. Please convert files first."
            )
        
        # Create a zip file in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add each transcription to the zip
            for file_id, file_info in storage.batch_files.items():
                if file_info["status"] == "transcribed" and file_info.get("transcribed_text"):
                    # Original filename without extension + .txt
                    original_name = Path(file_info["original_name"]).stem
                    zip_filename = f"{original_name}_transcript.txt"
                    
                    # Add file to zip
                    zip_file.writestr(zip_filename, file_info["transcribed_text"])
            
            # Add a summary file
            summary = f"Batch Transcription Summary\n"
            summary += f"Batch ID: {storage.batch_id}\n"
            summary += f"Total Files: {len(storage.batch_files)}\n"
            summary += f"Transcribed Files: {len(transcribed_files)}\n\n"
            
            for file_id, file_info in storage.batch_files.items():
                summary += f"File: {file_info['original_name']}\n"
                summary += f"Status: {file_info['status']}\n"
                if file_info.get("error"):
                    summary += f"Error: {file_info['error']}\n"
                summary += "\n"
                
            zip_file.writestr("batch_summary.txt", summary)
        
        # Reset buffer position
        zip_buffer.seek(0)
        
        # Create a filename for the zip
        zip_filename = f"transcriptions_batch_{storage.batch_id[:8]}.zip"
        
        # Return the zip file
        return FileResponse(
            path=None,
            media_type='application/zip',
            filename=zip_filename,
            content=zip_buffer.getvalue()
        )
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during batch download: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while preparing batch download: {str(e)}"
        )

@router.get("/batch-status", status_code=status.HTTP_200_OK)
async def get_batch_status(storage: FileStorage = Depends(get_file_storage)):
    """
    Get status information about the current batch.
    
    Args:
        storage: The FileStorage dependency with batch info
        
    Returns:
        JSON response with batch status details
    """
    try:
        # Check if batch exists
        if not storage.batch_id or not storage.batch_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No batch found. Please upload files first."
            )
            
        # Count files by status
        status_counts = {
            "uploaded": 0,
            "transcribed": 0,
            "error": 0
        }
        
        for file_info in storage.batch_files.values():
            status = file_info.get("status", "unknown")
            if status in status_counts:
                status_counts[status] += 1
        
        # Create file details
        file_details = []
        for file_id, file_info in storage.batch_files.items():
            file_details.append({
                "file_id": file_id,
                "original_name": file_info["original_name"],
                "status": file_info.get("status", "unknown"),
                "has_transcription": file_info.get("transcribed_text") is not None,
                "error": file_info.get("error")
            })
            
        return JSONResponse(content={
            "batch_id": storage.batch_id,
            "total_files": len(storage.batch_files),
            "status_counts": status_counts,
            "files": file_details
        })
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving batch status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while retrieving batch status: {str(e)}"
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

@router.delete("/batch-cleanup", status_code=status.HTTP_200_OK)
async def cleanup_batch(storage: FileStorage = Depends(get_file_storage)):
    """
    Clean up all resources associated with the current batch.
    
    Args:
        storage: The FileStorage dependency with batch info
        
    Returns:
        JSON confirmation of batch cleanup
    """
    try:
        # Check if batch exists
        if not storage.batch_id or not storage.batch_files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No batch found to clean up."
            )
            
        # Remove all files associated with the batch
        deleted_count = 0
        for file_id, file_info in storage.batch_files.items():
            try:
                # Delete source audio file
                if "path" in file_info and Path(file_info["path"]).is_file():
                    Path(file_info["path"]).unlink()
                    deleted_count += 1
                
                # Delete output transcription file if it exists
                if "output_path" in file_info and Path(file_info["output_path"]).is_file():
                    Path(file_info["output_path"]).unlink()
            except Exception as e:
                logger.warning(f"Error deleting file {file_id}: {str(e)}")
        
        # Reset batch info
        batch_id = storage.batch_id
        file_count = len(storage.batch_files)
        
        storage.batch_id = None
        storage.batch_files = {}
        
        logger.info(f"Batch {batch_id} cleaned up with {deleted_count} files deleted")
        
        return JSONResponse(content={
            "message": f"Batch cleaned up successfully",
            "batch_id": batch_id,
            "files_deleted": deleted_count,
            "total_files": file_count
        })
        
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error during batch cleanup: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during batch cleanup: {str(e)}"
        )
