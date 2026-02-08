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
from documentindex.models import DocumentType


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


class TestEarningsCallChunking:
    """Tests for earnings call document chunking with speaker patterns."""

    EARNINGS_CALL_TEXT = """## Q3 2024 Earnings Call Transcript

**Operator:**

Good morning and welcome to the Q3 2024 earnings conference call.
I would now like to turn the call over to Jane Smith, Head of Investor Relations.

**Jane Smith:**

Thank you, operator. Good morning everyone and welcome to our third quarter
2024 earnings call. Before we begin, I want to remind you that today's
discussion will contain forward-looking statements.

**John Doe:**

Thanks Jane. We had an outstanding quarter with revenue of $15.2 billion,
up 12% year over year. Operating margins expanded 150 basis points to 28.5%.
Let me walk you through the key highlights.

Our cloud segment grew 25% to $6.1 billion. Enterprise adoption accelerated
with over 200 new large deals signed in the quarter.

**Sarah Johnson:**

Thank you, John. Turning to the financials in more detail. Total revenue was
$15.2 billion, representing 12% growth. Gross margin was 65.3%, up from 63.8%
in the prior year quarter. Operating expenses were well managed.

Capital expenditures were $2.1 billion in the quarter, primarily directed toward
data center expansion and AI infrastructure investments.

Q&A Session

**Analyst Mike Brown:**

Great quarter. Can you talk about the CapEx outlook for 2025 and how much
of that is directed toward AI infrastructure?

**John Doe:**

Sure Mike. We expect CapEx to be in the range of $9 to $10 billion for 2025,
with approximately 60% directed toward AI and cloud infrastructure.
"""

    def test_earnings_call_no_speaker_sections(self):
        """set_doc_type(EARNINGS_CALL) should NOT produce per-speaker section titles."""
        chunker = TextChunker()
        chunker.set_doc_type(DocumentType.EARNINGS_CALL)
        chunks = chunker.chunk(self.EARNINGS_CALL_TEXT)

        bold_speaker_sections = [
            c.section_title for c in chunks
            if c.section_title and c.section_title.startswith("**")
        ]
        assert len(bold_speaker_sections) == 0, (
            f"Expected no bold-speaker sections, got {bold_speaker_sections}"
        )

    def test_default_chunker_misses_speakers(self):
        """Without set_doc_type, the default SEC patterns should NOT detect speakers."""
        chunker = TextChunker()
        chunks = chunker.chunk(self.EARNINGS_CALL_TEXT)

        # Default patterns will detect the markdown ## header but not **Speaker:** lines
        bold_speaker_sections = [
            c.section_title for c in chunks
            if c.section_title and c.section_title.startswith("**")
        ]
        assert len(bold_speaker_sections) == 0

    def test_earnings_call_produces_multiple_chunks(self):
        """Earnings call text should produce multiple chunks when using small config."""
        config = ChunkConfig(max_chunk_tokens=200)
        chunker = TextChunker(config)
        chunker.set_doc_type(DocumentType.EARNINGS_CALL)
        chunks = chunker.chunk(self.EARNINGS_CALL_TEXT)

        assert len(chunks) > 1, "Earnings call should produce multiple chunks"

    def test_set_doc_type_preserves_sec_patterns(self):
        """set_doc_type should merge type-specific patterns with SEC defaults."""
        chunker = TextChunker()
        chunker.set_doc_type(DocumentType.EARNINGS_CALL)

        # The merged patterns should still detect SEC-style PART headers
        text_with_part = "PART I\n\nSome content here.\n\n**Speaker Name:**\n\nMore content."
        chunks = chunker.chunk(text_with_part)

        section_titles = [c.section_title for c in chunks if c.section_title]
        assert any("PART" in t for t in section_titles), (
            f"SEC PART pattern should still work after set_doc_type: {section_titles}"
        )

    def test_earnings_call_size_based_chunking(self):
        """A longer transcript should produce multiple size-based chunks."""
        # Build a transcript large enough to exceed a single chunk
        speakers = ["**CEO:**", "**CFO:**", "**Analyst 1:**", "**CEO:**", "**Analyst 2:**"]
        long_text = "## Q4 2024 Earnings Call Transcript\n\n"
        for speaker in speakers:
            long_text += f"{speaker}\n\n"
            # ~6000 chars per speaker turn ≈ 30K+ total
            long_text += ("This quarter we saw strong performance across all segments. " * 60) + "\n\n"

        config = ChunkConfig(max_chunk_tokens=2000, overlap_tokens=100)
        chunker = TextChunker(config)
        chunker.set_doc_type(DocumentType.EARNINGS_CALL)
        chunks = chunker.chunk(long_text)

        assert 3 <= len(chunks) <= 10, (
            f"Expected 3-10 size-based chunks, got {len(chunks)}"
        )

    def test_earnings_call_explicit_qa_header_detected(self):
        """A transcript with an explicit Q&A header should still split on it."""
        text = (
            "Operator\n\nWelcome to the call.\n\n"
            "Prepared remarks content here. " * 20 + "\n\n"
            "Q&A\n\n"
            "Analyst questions and answers. " * 20 + "\n\n"
        )
        chunker = TextChunker()
        chunker.set_doc_type(DocumentType.EARNINGS_CALL)
        chunks = chunker.chunk(text)

        section_titles = [c.section_title for c in chunks if c.section_title]
        assert any("Q&A" in t or "Q & A" in t for t in section_titles), (
            f"Expected Q&A section header to be detected: {section_titles}"
        )


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
