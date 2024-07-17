# Start by making sure the `assemblyai` package is installed.
# If not, you can install it by running the following command:
# pip install -U assemblyai
#
# Note: Some macOS users may need to use `pip3` instead of `pip`.

import assemblyai as aai

# Replace with your API key
aai.settings.api_key = "cbfff14c51c5456a9d599939e886a7e6"

# URL of the file to transcribe
FILE_URL = "C:\\Users\\ESHOP\\Documents\\Share Mobility\\speech-to-text\\test_audio.mp3"

# You can set additional parameters for the transcription
config = aai.TranscriptionConfig(
  speech_model=aai.SpeechModel.best,
  sentiment_analysis=True,
  entity_detection=True,
  speaker_labels=True,
  speakers_expected=3,
  language_code="en_us"
)
    


# You can also transcribe a local file by passing in a file path
# FILE_URL = './path/to/file.mp3'

transcriber = aai.Transcriber(config=config)
transcript = transcriber.transcribe(FILE_URL)

# if transcript.status == aai.TranscriptStatus.error:
#     print(transcript.error)
# else:
#     print(transcript.text)
  
 
for utterance in transcript.utterances:
  print(f"Speaker {utterance.speaker}: {utterance.text}")    
  