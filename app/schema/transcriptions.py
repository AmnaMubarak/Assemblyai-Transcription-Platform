from pydantic import BaseModel, Field, validator
import assemblyai as aai
from typing import Optional, Dict, Any, Union
from enum import Enum

class SpeechModel(str, Enum):
    """Enum for supported speech models"""
    STANDARD = "standard"
    BEST = "best"

class LanguageCode(str, Enum):
    """Enum for supported language codes"""
    EN_US = "en_us"
    EN_AU = "en_au"
    EN_UK = "en_uk"
    FR = "fr"
    DE = "de"
    ES = "es"
    # Add other languages as needed

class TranscriptionConfig(BaseModel):
    """
    Configuration settings for the AssemblyAI transcription service.
    
    Attributes:
        speech_model: The speech model to use for transcription
        sentiment_analysis: Enable sentiment analysis
        entity_detection: Enable entity detection
        speaker_labels: Enable speaker diarization
        speakers_expected: Number of expected speakers (when speaker_labels is True)
        language_code: Language code for the audio content
    """
    speech_model: Union[aai.SpeechModel, SpeechModel] = Field(
        default=aai.SpeechModel.standard,
        description="Speech recognition model to use"
    )
    sentiment_analysis: bool = Field(
        default=False,
        description="Enable sentiment analysis"
    )
    entity_detection: bool = Field(
        default=False,
        description="Enable named entity detection"
    )
    speaker_labels: bool = Field(
        default=False,
        description="Enable speaker diarization"
    )
    speakers_expected: Optional[int] = Field(
        default=None,
        description="Number of speakers expected in the audio"
    )
    language_code: Union[str, LanguageCode] = Field(
        default=LanguageCode.EN_US,
        description="Language code for the audio content"
    )
    
    @validator('speakers_expected')
    def validate_speakers(cls, v, values):
        """Validate that speakers_expected is set when speaker_labels is True"""
        if values.get('speaker_labels') and (v is None or v < 1):
            raise ValueError("Number of expected speakers must be set when speaker_labels is True")
        return v
    
    class Config:
        """Pydantic model configuration"""
        use_enum_values = True
        arbitrary_types_allowed = True
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary format for the AssemblyAI API"""
        result = self.dict(exclude_none=True)
        
        # Convert speech_model if it's an enum value
        if isinstance(self.speech_model, SpeechModel):
            if self.speech_model == SpeechModel.BEST:
                result['speech_model'] = aai.SpeechModel.best
            else:
                result['speech_model'] = aai.SpeechModel.standard
                
        return result