# Start by making sure the `assemblyai` package is installed.
# If not, you can install it by running the following command:
# pip install -U assemblyai
#
# Note: Some macOS users may need to use `pip3` instead of `pip`.

import assemblyai as aai

# Replace with your API key
# aai.settings.api_key = "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"

# URL of the file to transcribe
FILE_URL = "C:\\Users\\ESHOP\\Documents\\Share Mobility\\speech-to-text\\test_audio.mp3"

aai.settings.api_key = "Your_API_Key"
transcriber = aai.Transcriber()
config = aai.TranscriptionConfig(
  speech_model=aai.SpeechModel.best,
  speaker_labels=True,
  language_detection=True
)
    
# You can also transcribe a local file by passing in a file path
# FILE_URL = './path/to/file.mp3'

transcriber = aai.Transcriber(config=config)
transcript = transcriber.transcribe(FILE_URL)

if transcript.status == aai.TranscriptStatus.error:
    print(transcript.error)
else:
    print(transcript.text)
  
  