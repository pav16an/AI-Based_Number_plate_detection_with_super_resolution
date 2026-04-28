"""
Validation utilities for license plate detection system
"""

import re
from typing import Tuple


class LicensePlateValidator:
    """Validator for license plate text"""
    
    # Common OCR errors to reject
    INVALID_PATTERNS = {
        'III', '000', 'OOO', 'LLL', '111',
        'llll', 'iiii', 'oooo', 'EEEE'
    }
    
    # Minimum and maximum plate length
    MIN_LENGTH = 3
    MAX_LENGTH = 10
    
    @classmethod
    def validate(cls, text: str) -> Tuple[bool, str]:
        """
        Validate license plate text
        
        Args:
            text: License plate text to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not text:
            return False, "Empty text"
        
        text = text.strip().upper()
        
        # Check length
        if len(text) < cls.MIN_LENGTH:
            return False, f"Text too short (min {cls.MIN_LENGTH} chars)"
        
        if len(text) > cls.MAX_LENGTH:
            return False, f"Text too long (max {cls.MAX_LENGTH} chars)"
        
        # Must contain at least one digit
        if not any(c.isdigit() for c in text):
            return False, "Must contain at least one digit"
        
        # Must contain at least one letter
        if not any(c.isalpha() for c in text):
            return False, "Must contain at least one letter"
        
        # Check for invalid patterns
        if text in cls.INVALID_PATTERNS:
            return False, f"Invalid pattern: {text}"
        
        # Check for suspicious repetitions
        if cls._has_suspicious_repetitions(text):
            return False, "Suspicious character repetitions"
        
        return True, "Valid"
    
    @staticmethod
    def _has_suspicious_repetitions(text: str, threshold: int = 3) -> bool:
        """Check if text has suspicious character repetitions"""
        for i in range(len(text) - threshold + 1):
            if len(set(text[i:i+threshold])) == 1:
                return True
        return False
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean OCR text to valid characters only"""
        if not text:
            return ""
        # Keep only alphanumeric characters
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
        return cleaned


class ImageValidator:
    """Validator for uploaded images"""
    
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    MIN_WIDTH = 100
    MIN_HEIGHT = 50
    MAX_WIDTH = 4096
    MAX_HEIGHT = 4096
    
    @classmethod
    def validate_extension(cls, filename: str) -> Tuple[bool, str]:
        """Validate file extension"""
        if '.' not in filename:
            return False, "No file extension"
        
        ext = filename.rsplit('.', 1)[1].lower()
        if ext not in cls.ALLOWED_EXTENSIONS:
            return False, f"Invalid extension. Allowed: {', '.join(cls.ALLOWED_EXTENSIONS)}"
        
        return True, "Valid extension"
    
    @classmethod
    def validate_dimensions(cls, width: int, height: int) -> Tuple[bool, str]:
        """Validate image dimensions"""
        if width < cls.MIN_WIDTH or height < cls.MIN_HEIGHT:
            return False, f"Image too small (min {cls.MIN_WIDTH}x{cls.MIN_HEIGHT})"
        
        if width > cls.MAX_WIDTH or height > cls.MAX_HEIGHT:
            return False, f"Image too large (max {cls.MAX_WIDTH}x{cls.MAX_HEIGHT})"
        
        return True, "Valid dimensions"


class VideoValidator:
    """Validator for video files"""
    
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv', 'wmv'}
    MAX_DURATION = 3600  # 1 hour in seconds
    MIN_FPS = 15
    
    @classmethod
    def validate_extension(cls, filename: str) -> Tuple[bool, str]:
        """Validate video file extension"""
        if '.' not in filename:
            return False, "No file extension"
        
        ext = filename.rsplit('.', 1)[1].lower()
        if ext not in cls.ALLOWED_EXTENSIONS:
            return False, f"Invalid extension. Allowed: {', '.join(cls.ALLOWED_EXTENSIONS)}"
        
        return True, "Valid extension"
