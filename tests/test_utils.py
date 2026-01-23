"""
Tests for utility functions.
"""

import pytest
from datetime import datetime

from documentindex.utils import (
    generate_doc_id,
    truncate_text,
    truncate_middle,
    clean_text,
    extract_sentences,
    format_number,
    parse_date,
    roman_to_int,
    int_to_roman,
    estimate_tokens,
    chunk_text_simple,
    find_best_match,
    highlight_text,
)


class TestGenerateDocId:
    """Tests for document ID generation"""
    
    def test_generates_id(self):
        doc_id = generate_doc_id("Some document content")
        assert len(doc_id) > 0
    
    def test_consistent_for_same_content(self):
        content = "Test content"
        id1 = generate_doc_id(content)
        id2 = generate_doc_id(content)
        assert id1 == id2
    
    def test_different_for_different_content(self):
        id1 = generate_doc_id("Content 1")
        id2 = generate_doc_id("Content 2")
        assert id1 != id2
    
    def test_with_prefix(self):
        doc_id = generate_doc_id("Content", prefix="10K")
        assert doc_id.startswith("10K_")


class TestTruncateText:
    """Tests for text truncation"""
    
    def test_no_truncation_needed(self):
        text = "Short text"
        result = truncate_text(text, 100)
        assert result == text
    
    def test_truncates_with_suffix(self):
        text = "This is a longer piece of text that needs truncation"
        result = truncate_text(text, 20)
        assert len(result) == 20
        assert result.endswith("...")
    
    def test_custom_suffix(self):
        text = "This is a longer piece of text"
        result = truncate_text(text, 15, suffix="[...]")
        assert result.endswith("[...]")


class TestTruncateMiddle:
    """Tests for middle truncation"""
    
    def test_no_truncation_needed(self):
        text = "Short text"
        result = truncate_middle(text, 100)
        assert result == text
    
    def test_truncates_middle(self):
        text = "Beginning of text" + " middle " * 100 + "End of text"
        result = truncate_middle(text, 100)
        
        assert len(result) <= 100 + 20  # Allow for truncation marker
        assert "Beginning" in result
        assert "End" in result
        assert "[truncated]" in result


class TestCleanText:
    """Tests for text cleaning"""
    
    def test_normalize_newlines(self):
        text = "Line 1\r\nLine 2\rLine 3\nLine 4"
        result = clean_text(text)
        assert "\r" not in result
    
    def test_remove_excessive_newlines(self):
        text = "Para 1\n\n\n\n\nPara 2"
        result = clean_text(text)
        assert "\n\n\n" not in result
    
    def test_strip_whitespace(self):
        text = "  Some text with spaces  "
        result = clean_text(text)
        assert not result.startswith(" ")
        assert not result.endswith(" ")


class TestExtractSentences:
    """Tests for sentence extraction"""
    
    def test_extract_first_sentences(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        sentences = extract_sentences(text, max_sentences=2)
        assert len(sentences) == 2
        assert sentences[0] == "First sentence."
    
    def test_handles_question_marks(self):
        text = "Is this a question? Yes, it is. And an exclamation!"
        sentences = extract_sentences(text, max_sentences=3)
        assert len(sentences) == 3


class TestFormatNumber:
    """Tests for number formatting"""
    
    def test_format_billions(self):
        assert "B" in format_number(1_500_000_000)
        assert "1.50B" == format_number(1_500_000_000)
    
    def test_format_millions(self):
        assert "M" in format_number(5_000_000)
        assert "5.00M" == format_number(5_000_000)
    
    def test_format_thousands(self):
        assert "K" in format_number(10_000)
        assert "10.00K" == format_number(10_000)
    
    def test_format_small(self):
        result = format_number(500)
        assert "K" not in result
        assert "500" in result


class TestParseDate:
    """Tests for date parsing"""
    
    def test_parse_iso_format(self):
        result = parse_date("2024-01-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15
    
    def test_parse_us_format(self):
        result = parse_date("01/15/2024")
        assert result is not None
        assert result.year == 2024
    
    def test_parse_long_format(self):
        result = parse_date("January 15, 2024")
        assert result is not None
        assert result.month == 1
    
    def test_parse_short_month(self):
        result = parse_date("Jan 15, 2024")
        assert result is not None
        assert result.month == 1
    
    def test_invalid_date(self):
        result = parse_date("not a date")
        assert result is None


class TestRomanNumerals:
    """Tests for Roman numeral conversion"""
    
    def test_roman_to_int_basic(self):
        assert roman_to_int("I") == 1
        assert roman_to_int("V") == 5
        assert roman_to_int("X") == 10
        assert roman_to_int("L") == 50
        assert roman_to_int("C") == 100
    
    def test_roman_to_int_compound(self):
        assert roman_to_int("IV") == 4
        assert roman_to_int("IX") == 9
        assert roman_to_int("XIV") == 14
        assert roman_to_int("XIX") == 19
        assert roman_to_int("MCMXCIV") == 1994
    
    def test_roman_to_int_lowercase(self):
        assert roman_to_int("iv") == 4
        assert roman_to_int("xix") == 19
    
    def test_int_to_roman_basic(self):
        assert int_to_roman(1) == "I"
        assert int_to_roman(5) == "V"
        assert int_to_roman(10) == "X"
    
    def test_int_to_roman_compound(self):
        assert int_to_roman(4) == "IV"
        assert int_to_roman(9) == "IX"
        assert int_to_roman(14) == "XIV"
        assert int_to_roman(1994) == "MCMXCIV"
    
    def test_int_to_roman_invalid(self):
        assert int_to_roman(0) == "0"
        assert int_to_roman(-5) == "-5"


class TestEstimateTokens:
    """Tests for token estimation"""
    
    def test_empty_string(self):
        assert estimate_tokens("") == 0
    
    def test_estimate(self):
        text = "Hello, world!"  # 13 chars
        tokens = estimate_tokens(text)
        assert tokens > 0
        assert tokens < len(text)


class TestChunkTextSimple:
    """Tests for simple text chunking"""
    
    def test_no_chunking_needed(self):
        text = "Short text"
        chunks = chunk_text_simple(text, max_chars=100)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunks_long_text(self):
        text = "A" * 500
        chunks = chunk_text_simple(text, max_chars=100, overlap=10)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 110  # Allow some tolerance
    
    def test_preserves_content(self):
        text = "Word1 Word2 Word3 Word4 Word5 " * 20
        chunks = chunk_text_simple(text, max_chars=50, overlap=0)
        
        # Verify all words appear somewhere
        all_text = " ".join(chunks)
        for word in ["Word1", "Word2", "Word3", "Word4", "Word5"]:
            assert word in all_text


class TestFindBestMatch:
    """Tests for fuzzy matching"""
    
    def test_exact_match(self):
        candidates = ["apple", "banana", "cherry"]
        result = find_best_match("apple", candidates)
        assert result == "apple"
    
    def test_partial_match(self):
        candidates = ["climate change risks", "regulatory compliance", "revenue growth"]
        result = find_best_match("climate risks", candidates)
        assert result == "climate change risks"
    
    def test_no_match(self):
        candidates = ["apple", "banana", "cherry"]
        result = find_best_match("completely different", candidates)
        assert result is None
    
    def test_empty_candidates(self):
        result = find_best_match("query", [])
        assert result is None


class TestHighlightText:
    """Tests for text highlighting"""
    
    def test_highlight_single_term(self):
        text = "The quick brown fox"
        result = highlight_text(text, ["quick"])
        assert "**quick**" in result
    
    def test_highlight_multiple_terms(self):
        text = "The quick brown fox jumps"
        result = highlight_text(text, ["quick", "fox"])
        assert "**quick**" in result
        assert "**fox**" in result
    
    def test_custom_markers(self):
        text = "The quick brown fox"
        result = highlight_text(text, ["quick"], before="[", after="]")
        assert "[quick]" in result
    
    def test_case_insensitive(self):
        text = "The QUICK brown fox"
        result = highlight_text(text, ["quick"])
        # Should match case-insensitively
        assert "**" in result
