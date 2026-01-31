"""Simple sentence splitter using regex."""
import re
from typing import List


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex, preserving punctuation.
    
    Handles common sentence endings: . ! ? followed by space or end of string.
    """
    # Pattern: sentence ending (. ! ?) followed by space or end of string
    # Use findall to preserve delimiters
    pattern = r'[^.!?]+[.!?]+(?:\s+|$)'
    sentences = re.findall(pattern, text)
    # Also handle text without sentence-ending punctuation
    if not sentences:
        # If no sentence endings found, return the whole text
        return [text.strip()] if text.strip() else []
    # Strip whitespace and filter empty
    return [s.strip() for s in sentences if s.strip()]
