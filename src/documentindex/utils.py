"""
Utility functions for DocumentIndex.
"""

import hashlib
import re
from typing import Optional
from datetime import datetime


def generate_doc_id(text: str, prefix: str = "") -> str:
    """
    Generate a unique document ID from text content.
    
    Args:
        text: Document text
        prefix: Optional prefix for the ID
    
    Returns:
        Unique ID string
    """
    hash_val = hashlib.sha256(text.encode()).hexdigest()[:12]
    if prefix:
        return f"{prefix}_{hash_val}"
    return hash_val


def truncate_text(text: str, max_chars: int, suffix: str = "...") -> str:
    """
    Truncate text to max characters with suffix.
    
    Args:
        text: Text to truncate
        max_chars: Maximum characters
        suffix: Suffix to add when truncated
    
    Returns:
        Truncated text
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def truncate_middle(text: str, max_chars: int) -> str:
    """
    Truncate text from the middle, keeping beginning and end.
    
    Args:
        text: Text to truncate
        max_chars: Maximum total characters
    
    Returns:
        Truncated text with middle replaced by ellipsis
    """
    if len(text) <= max_chars:
        return text
    
    keep = (max_chars - 20) // 2  # Reserve space for ellipsis
    return text[:keep] + "\n...[truncated]...\n" + text[-keep:]


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    - Normalize whitespace
    - Remove excessive newlines
    - Strip leading/trailing whitespace
    
    Args:
        text: Text to clean
    
    Returns:
        Cleaned text
    """
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive blank lines (more than 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Normalize spaces (but preserve single newlines)
    lines = text.split('\n')
    lines = [' '.join(line.split()) for line in lines]
    text = '\n'.join(lines)
    
    return text.strip()


def extract_sentences(text: str, max_sentences: int = 5) -> list[str]:
    """
    Extract first N sentences from text.
    
    Args:
        text: Source text
        max_sentences: Maximum sentences to extract
    
    Returns:
        List of sentences
    """
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return sentences[:max_sentences]


def format_number(num: float, precision: int = 2) -> str:
    """
    Format number with appropriate suffix (K, M, B).
    
    Args:
        num: Number to format
        precision: Decimal places
    
    Returns:
        Formatted string
    """
    if abs(num) >= 1e9:
        return f"{num / 1e9:.{precision}f}B"
    elif abs(num) >= 1e6:
        return f"{num / 1e6:.{precision}f}M"
    elif abs(num) >= 1e3:
        return f"{num / 1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse various date formats.
    
    Supports:
    - 2024-01-15
    - January 15, 2024
    - Jan 15, 2024
    - 01/15/2024
    - 15-Jan-2024
    
    Args:
        date_str: Date string to parse
    
    Returns:
        datetime or None if parsing fails
    """
    formats = [
        "%Y-%m-%d",
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%d-%b-%Y",
        "%d %B %Y",
    ]
    
    date_str = date_str.strip()
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None


def roman_to_int(s: str) -> int:
    """
    Convert Roman numeral to integer.
    
    Args:
        s: Roman numeral string (e.g., "IV", "XII")
    
    Returns:
        Integer value
    """
    roman_values = {
        'I': 1, 'V': 5, 'X': 10, 'L': 50,
        'C': 100, 'D': 500, 'M': 1000
    }
    
    s = s.upper().strip()
    result = 0
    prev = 0
    
    for char in reversed(s):
        curr = roman_values.get(char, 0)
        if curr < prev:
            result -= curr
        else:
            result += curr
        prev = curr
    
    return result


def int_to_roman(num: int) -> str:
    """
    Convert integer to Roman numeral.
    
    Args:
        num: Integer (1-3999)
    
    Returns:
        Roman numeral string
    """
    if num < 1 or num > 3999:
        return str(num)
    
    val_map = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
    ]
    
    result = []
    for value, numeral in val_map:
        while num >= value:
            result.append(numeral)
            num -= value
    
    return ''.join(result)


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.
    
    Uses simple heuristic: ~4 characters per token.
    
    Args:
        text: Text to estimate
    
    Returns:
        Estimated token count
    """
    return len(text) // 4


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens using tiktoken (if available).
    
    Falls back to estimate if tiktoken not installed.
    
    Args:
        text: Text to count
        model: Model name for encoding
    
    Returns:
        Token count
    """
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        return estimate_tokens(text)


def chunk_text_simple(text: str, max_chars: int, overlap: int = 100) -> list[str]:
    """
    Simple text chunking by character count.
    
    Args:
        text: Text to chunk
        max_chars: Maximum characters per chunk
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chars
        
        if end < len(text):
            # Find a good break point
            for sep in ['\n\n', '\n', '. ', ' ']:
                break_point = text.rfind(sep, start, end)
                if break_point > start + max_chars // 2:
                    end = break_point + len(sep)
                    break
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks


def find_best_match(query: str, candidates: list[str], threshold: float = 0.5) -> Optional[str]:
    """
    Find best matching string from candidates.
    
    Uses simple word overlap scoring.
    
    Args:
        query: Query string
        candidates: List of candidate strings
        threshold: Minimum match score (0-1)
    
    Returns:
        Best matching candidate or None
    """
    if not candidates:
        return None
    
    query_words = set(query.lower().split())
    best_match = None
    best_score = threshold
    
    for candidate in candidates:
        candidate_words = set(candidate.lower().split())
        
        if not candidate_words:
            continue
        
        # Jaccard similarity
        intersection = len(query_words & candidate_words)
        union = len(query_words | candidate_words)
        score = intersection / union if union > 0 else 0
        
        if score > best_score:
            best_score = score
            best_match = candidate
    
    return best_match


def highlight_text(text: str, terms: list[str], before: str = "**", after: str = "**") -> str:
    """
    Highlight terms in text.
    
    Args:
        text: Source text
        terms: Terms to highlight
        before: String to insert before match
        after: String to insert after match
    
    Returns:
        Text with highlighted terms
    """
    for term in terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        text = pattern.sub(f"{before}{term}{after}", text)
    return text
