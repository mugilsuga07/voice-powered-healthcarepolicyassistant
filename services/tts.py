"""
Text-to-Speech Service
Converts text responses to natural speech using OpenAI TTS API.
"""

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Result from text-to-speech synthesis."""
    audio_bytes: Optional[bytes]
    latency_ms: int
    success: bool
    error: Optional[str] = None


class TextToSpeechService:
    """
    Service for converting text to speech using OpenAI TTS.
    """
    
    # Available voices: alloy, echo, fable, onyx, nova, shimmer
    DEFAULT_VOICE = "nova"  # Professional, clear voice good for call centers
    DEFAULT_MODEL = "tts-1"  # Fast model; use "tts-1-hd" for higher quality
    
    def __init__(self):
        """Initialize the TTS service."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.client = OpenAI(api_key=api_key)
        self.voice = self.DEFAULT_VOICE
        self.model = self.DEFAULT_MODEL
    
    def synthesize(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0
    ) -> TTSResult:
        """
        Convert text to speech.
        
        Args:
            text: The text to convert to speech
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            speed: Speech speed (0.25 to 4.0, default 1.0)
            
        Returns:
            TTSResult with audio bytes and latency
        """
        start_time = time.time()
        
        if not text or not text.strip():
            return TTSResult(
                audio_bytes=None,
                latency_ms=0,
                success=False,
                error="No text provided"
            )
        
        try:
            # Use provided voice or default
            use_voice = voice or self.voice
            
            # Limit text length for TTS (OpenAI has a 4096 char limit)
            if len(text) > 4000:
                text = text[:4000] + "..."
                logger.warning("Text truncated for TTS (exceeded 4000 chars)")
            
            # Call OpenAI TTS API
            response = self.client.audio.speech.create(
                model=self.model,
                voice=use_voice,
                input=text,
                speed=speed,
                response_format="mp3"
            )
            
            # Get audio bytes
            audio_bytes = response.content
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            logger.info(f"TTS synthesis complete: {len(audio_bytes)} bytes in {latency_ms}ms")
            
            return TTSResult(
                audio_bytes=audio_bytes,
                latency_ms=latency_ms,
                success=True
            )
            
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"TTS error: {e}")
            return TTSResult(
                audio_bytes=None,
                latency_ms=latency_ms,
                success=False,
                error=str(e)
            )


# Global service instance
_tts_service = None


def get_tts_service() -> TextToSpeechService:
    """Get or create the global TTS service instance."""
    global _tts_service
    if _tts_service is None:
        _tts_service = TextToSpeechService()
    return _tts_service


def text_to_speech(
    text: str,
    voice: Optional[str] = None,
    speed: float = 1.0
) -> TTSResult:
    """
    Convenience function to convert text to speech.
    
    Args:
        text: The text to convert
        voice: Optional voice override
        speed: Speech speed (default 1.0)
        
    Returns:
        TTSResult with audio bytes and metadata
    """
    service = get_tts_service()
    return service.synthesize(text, voice=voice, speed=speed)

