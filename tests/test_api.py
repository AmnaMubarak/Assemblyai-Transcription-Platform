import pytest
from fastapi.testclient import TestClient
import os
from pathlib import Path
import tempfile

# Import after patching
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_api_root():
    """Test the API root returns the index.html file"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

class TestAudioAPI:
    """Tests for the audio API endpoints"""
    
    def test_upload_no_file(self):
        """Test the upload endpoint with no file"""
        response = client.post("/audio/upload")
        assert response.status_code == 422  # Validation error
    
    def test_upload_wrong_file_type(self):
        """Test the upload endpoint with wrong file type"""
        # Create a temporary text file
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
            temp_file.write(b"This is not an audio file")
            temp_file.flush()
            
            with open(temp_file.name, "rb") as f:
                response = client.post(
                    "/audio/upload",
                    files={"file": ("test.txt", f, "text/plain")}
                )
            
            assert response.status_code == 415
            assert "File must be an audio file" in response.json()["detail"]
    
    def test_convert_without_upload(self):
        """Test the convert endpoint without uploading first"""
        response = client.post("/audio/convert")
        assert response.status_code == 400
        assert "No file uploaded" in response.json()["detail"]
    
    def test_download_without_convert(self):
        """Test the download endpoint without converting first"""
        response = client.get("/audio/download")
        assert response.status_code == 400
        assert "No transcription available" in response.json()["detail"]

# Integration test requires a real audio file and API key
@pytest.mark.integration
def test_end_to_end_flow():
    """Test the end-to-end flow (requires a real audio file and API key)"""
    # Skip if no API key or not running in CI environment
    if not os.getenv("api_key") or not os.getenv("CI"):
        pytest.skip("Skipping integration test: No API key or not in CI environment")
    
    # Create a test audio file path
    test_audio_path = Path(__file__).parent / "test_audio.mp3"
    
    # Check if test audio file exists
    if not test_audio_path.exists():
        pytest.skip(f"Test audio file not found at {test_audio_path}")
    
    # Step 1: Upload the audio file
    with open(test_audio_path, "rb") as f:
        upload_response = client.post(
            "/audio/upload",
            files={"file": ("test_audio.mp3", f, "audio/mpeg")}
        )
    
    assert upload_response.status_code == 201
    assert "File uploaded successfully" in upload_response.json()["message"]
    
    # Step 2: Convert the audio to text
    convert_response = client.post("/audio/convert")
    assert convert_response.status_code == 200
    assert "Transcription successful" in convert_response.json()["message"]
    
    # Step 3: Download the transcription
    download_response = client.get("/audio/download")
    assert download_response.status_code == 200
    assert "Speaker" in download_response.text
    
    # Step 4: Clean up resources
    cleanup_response = client.delete("/audio/cleanup")
    assert cleanup_response.status_code == 200
    assert "Resources cleaned up successfully" in cleanup_response.json()["message"] 