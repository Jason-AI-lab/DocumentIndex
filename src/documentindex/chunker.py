"""
Text chunking with semantic boundary detection.

Splits text into chunks while respecting:
- Section headers (PART I, ITEM 1, etc.)
- Paragraph boundaries
- Sentence boundaries
- Token limits
"""

from dataclasses import dataclass, field
from typing import Optional
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for text chunking"""
    max_chunk_tokens: int = 2000
    overlap_tokens: int = 100
    min_chunk_tokens: int = 100
    
    # Approximate chars per token (conservative estimate)
    chars_per_token: float = 3.5
    
    # Whether to respect section boundaries
    respect_sections: bool = True
    respect_paragraphs: bool = True
    
    @property
    def max_chunk_chars(self) -> int:
        return int(self.max_chunk_tokens * self.chars_per_token)
    
    @property
    def overlap_chars(self) -> int:
        return int(self.overlap_tokens * self.chars_per_token)
    
    @property
    def min_chunk_chars(self) -> int:
        return int(self.min_chunk_tokens * self.chars_per_token)


@dataclass
class Chunk:
    """A chunk of text with position information"""
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    
    # Optional section info
    section_title: Optional[str] = None
    section_level: int = 0
    
    @property
    def char_count(self) -> int:
        return len(self.text)
    
    def __repr__(self) -> str:
        preview = self.text[:50].replace('\n', ' ')
        return f"Chunk({self.chunk_index}, {self.start_char}-{self.end_char}, '{preview}...')"


class TextChunker:
    """
    Chunks text while respecting semantic boundaries.
    
    Designed for financial documents with:
    - SEC filing section patterns (PART I, ITEM 1, etc.)
    - Markdown-style headers
    - Natural paragraph breaks
    """
    
    # Section header patterns for financial documents
    SEC_SECTION_PATTERNS = [
        # Part headers
        (r'^PART\s+([IVX]+|\d+)\s*$', 1),
        (r'^PART\s+([IVX]+|\d+)\s*[-–—]\s*', 1),
        
        # Item headers
        (r'^ITEM\s+(\d+[A-Z]?)\.?\s+', 2),
        (r'^Item\s+(\d+[A-Z]?)\.?\s+', 2),
        
        # Note headers (financial statements)
        (r'^NOTE\s+(\d+)\.?\s*[-–—:]?\s*', 2),
        (r'^Note\s+(\d+)\.?\s*[-–—:]?\s*', 2),
        
        # Exhibit headers
        (r'^EXHIBIT\s+(\d+(?:\.\d+)?)', 2),
        (r'^Exhibit\s+(\d+(?:\.\d+)?)', 2),
        
        # Appendix headers
        (r'^APPENDIX\s+([A-Z](?:\d+)?)', 2),
        (r'^Appendix\s+([A-Z](?:\d+)?)', 2),
        
        # Generic numbered sections
        (r'^(\d+)\.\s+[A-Z]', 2),
        (r'^(\d+\.\d+)\s+[A-Z]', 3),
        
        # Markdown-style headers
        (r'^#{1}\s+', 1),
        (r'^#{2}\s+', 2),
        (r'^#{3}\s+', 3),
        (r'^#{4}\s+', 4),
    ]
    
    def __init__(self, config: Optional[ChunkConfig] = None):
        self.config = config or ChunkConfig()
        self._compiled_patterns = [
            (re.compile(pattern, re.MULTILINE | re.IGNORECASE), level)
            for pattern, level in self.SEC_SECTION_PATTERNS
        ]
    
    def chunk(self, text: str) -> list[Chunk]:
        """
        Chunk text into semantic units.
        
        Args:
            text: Full document text
        
        Returns:
            List of Chunk objects with position information
        """
        if not text:
            return []
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # First, identify section boundaries
        if self.config.respect_sections:
            sections = self._find_sections(text)
        else:
            sections = [(0, len(text), None, 0)]
        
        # Chunk each section
        chunks: list[Chunk] = []
        chunk_index = 0
        
        for section_start, section_end, section_title, section_level in sections:
            section_text = text[section_start:section_end]
            
            # Split section into paragraph-respecting chunks
            section_chunks = self._chunk_section(
                section_text,
                section_start,
                chunk_index,
                section_title,
                section_level,
            )
            
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        return chunks
    
    def chunk_to_strings(self, text: str) -> list[str]:
        """Chunk text and return just the text strings"""
        return [chunk.text for chunk in self.chunk(text)]
    
    def get_chunk_offsets(self, text: str) -> list[tuple[int, int]]:
        """Get character offsets for each chunk"""
        return [(chunk.start_char, chunk.end_char) for chunk in self.chunk(text)]
    
    def _find_sections(self, text: str) -> list[tuple[int, int, Optional[str], int]]:
        """
        Find section boundaries in text.
        
        Returns list of (start, end, title, level) tuples.
        """
        # Find all section headers with their positions
        headers: list[tuple[int, str, int]] = []  # (position, title, level)
        
        for line_match in re.finditer(r'^.+$', text, re.MULTILINE):
            line = line_match.group()
            line_start = line_match.start()
            
            for pattern, level in self._compiled_patterns:
                if pattern.match(line):
                    # Extract title from the line
                    title = line.strip()
                    headers.append((line_start, title, level))
                    break
        
        # If no headers found, return entire text as one section
        if not headers:
            return [(0, len(text), None, 0)]
        
        # Build sections from headers
        sections: list[tuple[int, int, Optional[str], int]] = []
        
        # Add content before first header if any
        if headers[0][0] > 0:
            sections.append((0, headers[0][0], None, 0))
        
        # Add sections between headers
        for i, (start, title, level) in enumerate(headers):
            if i + 1 < len(headers):
                end = headers[i + 1][0]
            else:
                end = len(text)
            
            sections.append((start, end, title, level))
        
        return sections
    
    def _chunk_section(
        self,
        text: str,
        base_offset: int,
        start_index: int,
        section_title: Optional[str],
        section_level: int,
    ) -> list[Chunk]:
        """Chunk a section while respecting paragraph boundaries"""
        if not text.strip():
            return []
        
        max_chars = self.config.max_chunk_chars
        min_chars = self.config.min_chunk_chars
        overlap_chars = self.config.overlap_chars
        
        # If section fits in one chunk, return as-is
        if len(text) <= max_chars:
            return [Chunk(
                text=text,
                start_char=base_offset,
                end_char=base_offset + len(text),
                chunk_index=start_index,
                section_title=section_title,
                section_level=section_level,
            )]
        
        # Split into paragraphs
        if self.config.respect_paragraphs:
            # Split on double newlines or more
            paragraphs = re.split(r'\n\s*\n', text)
        else:
            paragraphs = [text]
        
        chunks: list[Chunk] = []
        current_chunk_text = ""
        current_chunk_start = 0
        chunk_index = start_index
        text_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Find paragraph position in original text
            para_pos = text.find(para, text_pos)
            if para_pos == -1:
                para_pos = text_pos
            
            # Check if adding this paragraph exceeds limit
            test_text = current_chunk_text + ("\n\n" if current_chunk_text else "") + para
            
            if len(test_text) > max_chars and current_chunk_text:
                # Save current chunk
                chunks.append(Chunk(
                    text=current_chunk_text,
                    start_char=base_offset + current_chunk_start,
                    end_char=base_offset + current_chunk_start + len(current_chunk_text),
                    chunk_index=chunk_index,
                    section_title=section_title,
                    section_level=section_level,
                ))
                chunk_index += 1
                
                # Start new chunk (with overlap if configured)
                if overlap_chars > 0 and len(current_chunk_text) > overlap_chars:
                    # Find good break point for overlap
                    overlap_start = self._find_break_point(
                        current_chunk_text,
                        len(current_chunk_text) - overlap_chars
                    )
                    overlap_text = current_chunk_text[overlap_start:]
                    current_chunk_text = overlap_text + "\n\n" + para
                    current_chunk_start = current_chunk_start + overlap_start
                else:
                    current_chunk_text = para
                    current_chunk_start = para_pos
            else:
                if not current_chunk_text:
                    current_chunk_start = para_pos
                current_chunk_text = test_text
            
            text_pos = para_pos + len(para)
        
        # Save final chunk
        if current_chunk_text:
            chunks.append(Chunk(
                text=current_chunk_text,
                start_char=base_offset + current_chunk_start,
                end_char=base_offset + current_chunk_start + len(current_chunk_text),
                chunk_index=chunk_index,
                section_title=section_title,
                section_level=section_level,
            ))
        
        # Handle very large paragraphs that exceed max size
        final_chunks: list[Chunk] = []
        for chunk in chunks:
            if len(chunk.text) > max_chars * 1.5:  # Allow some tolerance
                # Force split on sentences
                sub_chunks = self._force_split_chunk(chunk, start_index + len(final_chunks))
                final_chunks.extend(sub_chunks)
            else:
                chunk.chunk_index = start_index + len(final_chunks)
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _force_split_chunk(self, chunk: Chunk, start_index: int) -> list[Chunk]:
        """Force split an oversized chunk on sentence boundaries"""
        text = chunk.text
        max_chars = self.config.max_chunk_chars
        
        # Split on sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        sub_chunks: list[Chunk] = []
        current_text = ""
        current_start = 0
        text_pos = 0
        
        for sentence in sentences:
            sentence_pos = text.find(sentence, text_pos)
            if sentence_pos == -1:
                sentence_pos = text_pos
            
            if len(current_text) + len(sentence) > max_chars and current_text:
                # Save current
                sub_chunks.append(Chunk(
                    text=current_text,
                    start_char=chunk.start_char + current_start,
                    end_char=chunk.start_char + current_start + len(current_text),
                    chunk_index=start_index + len(sub_chunks),
                    section_title=chunk.section_title,
                    section_level=chunk.section_level,
                ))
                current_text = sentence
                current_start = sentence_pos
            else:
                if not current_text:
                    current_start = sentence_pos
                current_text += (" " if current_text else "") + sentence
            
            text_pos = sentence_pos + len(sentence)
        
        if current_text:
            sub_chunks.append(Chunk(
                text=current_text,
                start_char=chunk.start_char + current_start,
                end_char=chunk.start_char + current_start + len(current_text),
                chunk_index=start_index + len(sub_chunks),
                section_title=chunk.section_title,
                section_level=chunk.section_level,
            ))
        
        return sub_chunks if sub_chunks else [chunk]
    
    def _find_break_point(self, text: str, target_pos: int) -> int:
        """Find a good break point near target position"""
        # Look for paragraph break
        para_break = text.rfind('\n\n', 0, target_pos + 100)
        if para_break > target_pos - 200:
            return para_break + 2
        
        # Look for sentence break
        sentence_end = max(
            text.rfind('. ', 0, target_pos + 50),
            text.rfind('! ', 0, target_pos + 50),
            text.rfind('? ', 0, target_pos + 50),
        )
        if sentence_end > target_pos - 100:
            return sentence_end + 2
        
        # Look for any line break
        line_break = text.rfind('\n', 0, target_pos + 50)
        if line_break > target_pos - 100:
            return line_break + 1
        
        # Fall back to word break
        space = text.rfind(' ', 0, target_pos + 20)
        if space > target_pos - 50:
            return space + 1
        
        return target_pos


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken.
    
    Falls back to character-based estimate if tiktoken unavailable.
    """
    try:
        import tiktoken
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Fallback: estimate ~4 chars per token
        return len(text) // 4
