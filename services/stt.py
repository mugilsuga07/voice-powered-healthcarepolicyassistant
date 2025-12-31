"""
Speech-to-Text Service
Handles audio transcription using OpenAI Whisper API
"""

import os
import time
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class TranscriptionResult:
    """Result from speech-to-text transcription."""
    text: str
    latency_ms: int
    success: bool
    error: Optional[str] = None


class SpeechToTextService:
    """Service for converting speech to text using OpenAI Whisper."""
    
    def __init__(self):
        """Initialize the STT service."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.client = OpenAI(api_key=api_key)
        self.model = "whisper-1"
    
    def transcribe(self, audio_bytes: bytes, filename: str = "audio.wav") -> TranscriptionResult:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_bytes: Raw audio data
            filename: Original filename (used to determine format)
            
        Returns:
            TranscriptionResult with text and latency
        """
        start_time = time.time()
        
        try:
            # Write audio bytes to a temporary file
            # Whisper API needs a file-like object with a name
            suffix = Path(filename).suffix or ".wav"
            
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_file_path = tmp_file.name
            
            try:
                # Call Whisper API
                with open(tmp_file_path, "rb") as audio_file:
                    transcript = self.client.audio.transcriptions.create(
                        model=self.model,
                        file=audio_file,
                        response_format="text"
                    )
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                return TranscriptionResult(
                    text=transcript.strip(),
                    latency_ms=latency_ms,
                    success=True
                )
                
            finally:
                # Clean up temp file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return TranscriptionResult(
                text="",
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> TranscriptionResult:
    """
    Convenience function to transcribe audio.
    
    Args:
        audio_bytes: Raw audio data
        filename: Original filename
        
    Returns:
        TranscriptionResult with text and latency
    """
    service = SpeechToTextService()
    return service.transcribe(audio_bytes, filename)

