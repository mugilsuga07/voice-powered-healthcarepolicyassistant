"""
Transcript Sanitization Service
Detects and redacts sensitive identifiers before sending to RAG pipeline.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SanitizationResult:
    """Result from transcript sanitization."""
    original_text: str
    sanitized_text: str
    redactions: List[dict] = field(default_factory=list)
    was_redacted: bool = False


class TranscriptSanitizer:
    """
    Sanitizes transcripts by detecting and redacting sensitive identifiers.
    
    Redacts:
    - Social Security Numbers (XXX-XX-XXXX)
    - Policy/Member IDs (alphanumeric patterns)
    - Long numeric strings (6+ digits)
    - Phone numbers
    - Dates of birth patterns
    - Email addresses
    - Credit card numbers
    """
    
    # Redaction patterns with names and regex
    PATTERNS: List[Tuple[str, str, str]] = [
        # (pattern_name, regex, replacement)
        (
            "SSN",
            r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",
            "[SSN REDACTED]"
        ),
        (
            "PHONE",
            r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "[PHONE REDACTED]"
        ),
        (
            "EMAIL",
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "[EMAIL REDACTED]"
        ),
        (
            "CREDIT_CARD",
            r"\b(?:\d{4}[-.\s]?){3}\d{4}\b",
            "[CARD REDACTED]"
        ),
        (
            "DOB",
            r"\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])[-/](?:19|20)?\d{2}\b",
            "[DOB REDACTED]"
        ),
        (
            "POLICY_ID",
            r"\b[A-Z]{2,4}[-]?\d{6,12}\b",
            "[POLICY ID REDACTED]"
        ),
        (
            "MEMBER_ID",
            r"\b(?:member|id|policy|account)[\s:#]*[A-Za-z0-9]{6,15}\b",
            "[MEMBER ID REDACTED]"
        ),
        (
            "LONG_NUMBER",
            r"\b\d{6,}\b",
            "[ID NUMBER REDACTED]"
        ),
    ]
    
    def __init__(self):
        """Initialize the sanitizer with compiled patterns."""
        self.compiled_patterns = [
            (name, re.compile(pattern, re.IGNORECASE), replacement)
            for name, pattern, replacement in self.PATTERNS
        ]
    
    def sanitize(self, text: str) -> SanitizationResult:
        """
        Sanitize transcript by redacting sensitive identifiers.
        
        Args:
            text: Original transcript text
            
        Returns:
            SanitizationResult with sanitized text and redaction info
        """
        if not text or not text.strip():
            return SanitizationResult(
                original_text=text,
                sanitized_text=text,
                redactions=[],
                was_redacted=False
            )
        
        sanitized = text
        redactions = []
        
        for pattern_name, pattern, replacement in self.compiled_patterns:
            matches = pattern.findall(sanitized)
            
            if matches:
                for match in matches:
                    # Log each redaction
                    redaction_info = {
                        "type": pattern_name,
                        "original_length": len(match) if isinstance(match, str) else len(str(match)),
                        "replacement": replacement
                    }
                    redactions.append(redaction_info)
                    
                    logger.info(
                        f"REDACTION: Type={pattern_name}, "
                        f"Length={redaction_info['original_length']} chars"
                    )
                
                # Apply redaction
                sanitized = pattern.sub(replacement, sanitized)
        
        was_redacted = len(redactions) > 0
        
        if was_redacted:
            logger.warning(
                f"Transcript sanitized: {len(redactions)} redaction(s) applied"
            )
        
        return SanitizationResult(
            original_text=text,
            sanitized_text=sanitized,
            redactions=redactions,
            was_redacted=was_redacted
        )


# Global sanitizer instance
_sanitizer = None


def get_sanitizer() -> TranscriptSanitizer:
    """Get or create the global sanitizer instance."""
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = TranscriptSanitizer()
    return _sanitizer


def sanitize_transcript(text: str) -> SanitizationResult:
    """
    Convenience function to sanitize a transcript.
    
    Args:
        text: Original transcript text
        
    Returns:
        SanitizationResult with sanitized text and redaction info
    """
    sanitizer = get_sanitizer()
    return sanitizer.sanitize(text)

