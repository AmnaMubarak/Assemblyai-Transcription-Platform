from pydantic import BaseModel
import assemblyai as aai

class TranscriptionConfig(BaseModel):
    speech_model: aai.SpeechModel
    sentiment_analysis: bool
    entity_detection: bool
    speaker_labels: bool
    speakers_expected: int
    language_code: str
    def dict(self):
        return self.__dict__