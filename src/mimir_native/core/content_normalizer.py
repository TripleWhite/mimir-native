"""Content Normalizer - Standardize different content formats"""

import re
from typing import Dict, Any, Optional
from .memory_entry import ContentType


class ContentNormalizer:
    """
    Normalizes various content types into a standard format.
    
    Supports:
    - Text cleaning and formatting
    - Code normalization
    - Conversation structuring
    - Document parsing
    """
    
    def __init__(self):
        self.max_length = 10000  # Max content length
    
    def normalize(self, content: str, content_type: ContentType) -> str:
        """
        Normalize content based on type.
        
        Args:
            content: Raw content
            content_type: Type of content
            
        Returns:
            Normalized content
        """
        if content_type == ContentType.TEXT:
            return self._normalize_text(content)
        elif content_type == ContentType.CODE:
            return self._normalize_code(content)
        elif content_type == ContentType.CONVERSATION:
            return self._normalize_conversation(content)
        elif content_type == ContentType.DOCUMENT:
            return self._normalize_document(content)
        elif content_type == ContentType.IMAGE_DESCRIPTION:
            return self._normalize_image_desc(content)
        else:
            return self._normalize_text(content)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize plain text"""
        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        # Truncate if too long
        if len(text) > self.max_length:
            text = text[:self.max_length] + "... [truncated]"
        
        return text
    
    def _normalize_code(self, code: str) -> str:
        """Normalize code"""
        # Remove trailing whitespace
        lines = code.split('\n')
        lines = [line.rstrip() for line in lines]
        
        # Remove excessive blank lines
        result = []
        prev_blank = False
        for line in lines:
            is_blank = len(line.strip()) == 0
            if is_blank and prev_blank:
                continue
            result.append(line)
            prev_blank = is_blank
        
        code = '\n'.join(result)
        
        # Ensure trailing newline
        if not code.endswith('\n'):
            code += '\n'
        
        return code
    
    def _normalize_conversation(self, conversation: str) -> str:
        """Normalize conversation format"""
        # Standardize speaker labels
        conversation = re.sub(r'^(User|Human|Assistant|AI|Bot):\s*', 
                            lambda m: f"{m.group(1)}: ", 
                            conversation, 
                            flags=re.MULTILINE)
        
        # Ensure consistent formatting
        lines = conversation.split('\n')
        formatted = []
        
        for line in lines:
            line = line.strip()
            if line:
                formatted.append(line)
        
        return '\n'.join(formatted)
    
    def _normalize_document(self, document: str) -> str:
        """Normalize document"""
        # Remove non-printable characters
        document = re.sub(r'[^\x20-\x7E\n\s]', '', document)
        
        # Normalize line endings
        document = document.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive blank lines
        document = re.sub(r'\n{3,}', '\n\n', document)
        
        return document.strip()
    
    def _normalize_image_desc(self, description: str) -> str:
        """Normalize image description"""
        # Clean up the description
        desc = description.strip()
        
        # Remove phrases like "This image shows..." if present
        desc = re.sub(r'^(This image shows?|The image shows?|In this image,)\s*', 
                     '', desc, flags=re.IGNORECASE)
        
        return desc
    
    def extract_summary(self, content: str, max_chars: int = 200) -> str:
        """Extract a summary of content"""
        # Get first paragraph or first N characters
        paragraphs = content.split('\n\n')
        first_para = paragraphs[0] if paragraphs else content
        
        if len(first_para) > max_chars:
            return first_para[:max_chars].rsplit(' ', 1)[0] + "..."
        
        return first_para
