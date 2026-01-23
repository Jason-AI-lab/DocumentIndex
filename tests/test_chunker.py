"""
Tests for text chunking.
"""

import pytest

from documentindex.chunker import (
    ChunkConfig,
    Chunk,
    TextChunker,
    count_tokens,
)


class TestChunkConfig:
    """Tests for ChunkConfig"""
    
    def test_default_config(self):
        config = ChunkConfig()
        assert config.max_chunk_tokens == 2000
        assert config.overlap_tokens == 100
        assert config.respect_sections is True
    
    def test_custom_config(self):
        config = ChunkConfig(
            max_chunk_tokens=1000,
            overlap_tokens=50,
            respect_sections=False,
        )
        assert config.max_chunk_tokens == 1000
        assert config.max_chunk_chars > 0
    
    def test_char_calculations(self):
        config = ChunkConfig(max_chunk_tokens=1000, chars_per_token=4.0)
        assert config.max_chunk_chars == 4000


class TestChunk:
    """Tests for Chunk model"""
    
    def test_create_chunk(self):
        chunk = Chunk(
            text="This is test content.",
            start_char=0,
            end_char=21,
            chunk_index=0,
        )
        assert chunk.char_count == 21
        assert chunk.chunk_index == 0
    
    def test_chunk_with_section(self):
        chunk = Chunk(
            text="Content",
            start_char=100,
            end_char=107,
            chunk_index=5,
            section_title="PART I",
            section_level=1,
        )
        assert chunk.section_title == "PART I"
        assert chunk.section_level == 1


class TestTextChunker:
    """Tests for TextChunker"""
    
    def test_empty_text(self):
        chunker = TextChunker()
        chunks = chunker.chunk("")
        assert chunks == []
    
    def test_short_text(self):
        chunker = TextChunker()
        text = "This is a short text."
        chunks = chunker.chunk(text)
        assert len(chunks) == 1
        assert chunks[0].text == text
    
    def test_preserves_text(self):
        chunker = TextChunker()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk(text)
        
        # Combine all chunks and verify content is preserved
        combined = " ".join(c.text for c in chunks)
        for word in ["First", "Second", "Third", "paragraph"]:
            assert word in combined
    
    def test_section_detection(self):
        chunker = TextChunker()
        text = """PART I

This is part one content with details.

PART II

This is part two content with more details.
"""
        chunks = chunker.chunk(text)
        
        # Should detect section headers
        section_titles = [c.section_title for c in chunks if c.section_title]
        assert len(section_titles) >= 1
    
    def test_item_detection(self):
        chunker = TextChunker()
        text = """ITEM 1. BUSINESS

Business description here.

ITEM 1A. RISK FACTORS

Risk factors here.

ITEM 2. PROPERTIES

Properties description.
"""
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1
    
    def test_long_text_splits(self):
        config = ChunkConfig(max_chunk_tokens=100)  # Small chunks
        chunker = TextChunker(config)
        
        # Create long text
        text = ("This is a paragraph with multiple sentences. " * 10 + "\n\n") * 20
        chunks = chunker.chunk(text)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should respect size limit (approximately)
        max_chars = config.max_chunk_chars * 1.5  # Allow some tolerance
        for chunk in chunks:
            assert chunk.char_count <= max_chars
    
    def test_chunk_offsets(self):
        chunker = TextChunker()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk(text)
        
        for chunk in chunks:
            # Verify offsets point to correct text
            assert chunk.start_char >= 0
            assert chunk.end_char <= len(text)
            assert chunk.start_char < chunk.end_char
    
    def test_chunk_to_strings(self):
        chunker = TextChunker()
        text = "Short test text."
        strings = chunker.chunk_to_strings(text)
        
        assert isinstance(strings, list)
        assert all(isinstance(s, str) for s in strings)
    
    def test_get_chunk_offsets(self):
        chunker = TextChunker()
        text = "Short test text."
        offsets = chunker.get_chunk_offsets(text)
        
        assert isinstance(offsets, list)
        assert all(isinstance(o, tuple) and len(o) == 2 for o in offsets)
    
    def test_markdown_headers(self):
        chunker = TextChunker()
        text = """# Main Title

Introduction text.

## Section One

Section one content.

## Section Two

Section two content.

### Subsection

Subsection content.
"""
        chunks = chunker.chunk(text)
        assert len(chunks) >= 1


class TestCountTokens:
    """Tests for token counting"""
    
    def test_count_tokens_basic(self):
        text = "Hello, world!"
        count = count_tokens(text)
        assert count > 0
        assert count < len(text)  # Tokens should be fewer than characters
    
    def test_count_empty_string(self):
        assert count_tokens("") == 0
    
    def test_count_long_text(self):
        text = "The quick brown fox jumps over the lazy dog. " * 100
        count = count_tokens(text)
        assert count > 100  # Should have significant tokens
