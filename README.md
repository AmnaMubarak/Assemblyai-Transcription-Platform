# Speech to Text Conversion Service

A robust web application for converting speech audio files to text using AssemblyAI's powerful transcription API. This service provides a clean, intuitive user interface for audio file uploads and high-quality transcriptions with speaker diarization.

![Speech to Text](https://img.shields.io/badge/Speech_to_Text-Conversion-4a6fa5)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-009688)
![AssemblyAI](https://img.shields.io/badge/AssemblyAI-API-black)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)

## Features

- **Audio File Upload**: Support for various audio formats (MP3, WAV, M4A, etc.)
- **Secure File Handling**: Validation of file types and size limits
- **Speech-to-Text Conversion**: High-quality transcription using AssemblyAI
- **Speaker Diarization**: Automatically identifies different speakers in the audio
- **Sentiment Analysis**: Detects sentiment in transcribed text (via AssemblyAI)
- **Entity Detection**: Identifies named entities in the transcription
- **Interactive UI**: Modern, responsive interface with real-time progress updates
- **Downloadable Transcripts**: Easy export of transcription results
- **Docker Support**: Containerized deployment for consistent environments
- **Comprehensive Testing**: Unit and integration tests for reliable functionality
- **Security Features**: File validation, error handling, and secure API key management

## Technology Stack

- **Backend**: FastAPI, Python 3.10+
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **API Integration**: AssemblyAI for speech-to-text processing
- **Containerization**: Docker and Docker Compose
- **Testing**: Pytest

## Project Structure

```
speech-to-text-conversion/
├── app/                       # Application code
│   ├── api/                   # API endpoints
│   │   ├── api_v1/            # API version 1
│   │   │   └── endpoints/     # API endpoint implementations
│   │   └── router.py          # API router configuration
│   ├── logs/                  # Application logs
│   ├── schema/                # Data validation schemas
│   ├── static/                # Static frontend files
│   ├── uploads/               # Temporary storage for uploads
│   └── main.py                # Application entry point
├── tests/                     # Test suite
├── .env.example               # Example environment variables
├── .gitignore                 # Git ignore file
├── docker-compose.yml         # Docker Compose configuration
├── Dockerfile                 # Docker image definition
├── README                     # Project documentation
└── requirements.txt           # Python dependencies
```

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- [AssemblyAI API key](https://www.assemblyai.com/) (Sign up for free)
- Docker and Docker Compose (optional, for containerized deployment)

### Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd speech-to-text-conversion
   ```

2. **Create and configure environment variables**:
   ```bash
   # Create a .env file based on the example
   cp .env.example .env
   
   # Edit the .env file and add your AssemblyAI API key
   nano .env
   ```

3. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   
   # For Windows
   venv\Scripts\activate
   
   # For Unix or MacOS
   source venv/bin/activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Create necessary directories**:
   ```bash
   mkdir -p app/uploads app/logs
   ```

6. **Start the FastAPI server**:
   ```bash
   uvicorn app.main:app --reload
   ```

7. **Access the application**:
   Open your browser and navigate to `http://127.0.0.1:8000`

### Docker Deployment

1. **Clone the repository and configure environment variables**:
   ```bash
   git clone <repository-url>
   cd speech-to-text-conversion
   cp .env.example .env
   nano .env  # Add your AssemblyAI API key
   ```

2. **Build and start the Docker containers**:
   ```bash
   docker-compose up -d --build
   ```

3. **Access the application**:
   Open your browser and navigate to `http://localhost:8000`

4. **View logs**:
   ```bash
   docker-compose logs -f
   ```

5. **Stop the containers**:
   ```bash
   docker-compose down
   ```

## Environment Variables

Create a `.env` file in the project root with the following variables:

```
# Required
api_key=YOUR_ASSEMBLYAI_API_KEY_HERE

# Optional (defaults shown)
DEBUG=True
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
UPLOAD_DIR=./uploads
MAX_UPLOAD_SIZE=10485760  # 10MB in bytes
```

## API Documentation

Once the server is running, visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI API documentation.

### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main web interface |
| `/health` | GET | Health check endpoint |
| `/audio/upload` | POST | Upload an audio file |
| `/audio/convert` | POST | Convert uploaded audio to text |
| `/audio/download` | GET | Download the transcription |
| `/audio/cleanup` | DELETE | Clean up temporary files |

## Usage Guide

1. **Upload an Audio File**:
   - Click "Choose Audio File" to select your audio file.
   - Click "Upload" to upload the file to the server.

2. **Convert to Text**:
   - After a successful upload, click "Convert to Text".
   - Wait for the transcription process to complete (this may take a moment).

3. **View and Download**:
   - Once transcription is complete, the text will be displayed on the page.
   - Click "Download Transcript" to save the transcription as a text file.

4. **Reset**:
   - Click "Reset" to clear the current session and start again.

## Transcription Configuration

The service uses the following default settings for transcription:

- Speech model: Best quality
- Speaker diarization: Enabled (detects up to 3 speakers)
- Sentiment analysis: Enabled
- Entity detection: Enabled
- Language: English (US)

These settings can be modified in the `app/api/api_v1/endpoints/audio.py` file.

## Testing

Run the test suite to ensure everything is working correctly:

```bash
# Run all tests
pytest

# Run only unit tests (excluding integration tests)
pytest -k "not integration"

# Run with verbose output
pytest -v
```

### Testing with Docker

```bash
docker-compose run --rm app pytest
```

## Security Considerations

- **API Key Protection**: Store your AssemblyAI API key in the `.env` file, which should not be committed to version control.
- **File Validation**: The application validates file types and sizes to prevent abuse.
- **CORS Protection**: CORS settings are restrictive by default and configurable via environment variables.
- **Error Handling**: Comprehensive error handling to prevent information leakage.

## Troubleshooting

- **File Upload Issues**: Ensure your audio file is in a supported format and under the maximum size limit.
- **Transcription Failures**: Check the application logs in `app/logs/` for detailed error information.
- **API Key Issues**: Verify your AssemblyAI API key is correctly set in the `.env` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [AssemblyAI](https://www.assemblyai.com/) for providing the speech-to-text API
- [FastAPI](https://fastapi.tiangolo.com/) for the powerful web framework

