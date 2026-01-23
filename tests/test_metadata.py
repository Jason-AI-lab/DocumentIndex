"""
Tests for metadata extraction.
"""

import pytest
from datetime import datetime

from documentindex.metadata import (
    MetadataExtractor,
    MetadataExtractorConfig,
)
from documentindex.models import DocumentType


class TestMetadataExtractor:
    """Tests for metadata extraction"""
    
    def test_extract_sync_basic(self, sample_10k_text):
        extractor = MetadataExtractor(MetadataExtractorConfig(use_llm_extraction=False))
        metadata = extractor.extract_sync(sample_10k_text, DocumentType.SEC_10K)
        
        # Should find CIK
        assert metadata.cik == "0001234567"
        
        # Should find ticker
        assert metadata.ticker == "ACME"
    
    def test_extract_company_name(self):
        text = """
        Company Name: Test Corporation
        REGISTRANT: Another Company Inc.
        """
        extractor = MetadataExtractor(MetadataExtractorConfig(use_llm_extraction=False))
        metadata = extractor.extract_sync(text, DocumentType.GENERIC)
        
        # Should find company name
        assert metadata.company_name is not None
    
    def test_extract_filing_date(self):
        text = """
        Filing Date: January 15, 2024
        Filed: 2024-03-01
        """
        extractor = MetadataExtractor(MetadataExtractorConfig(use_llm_extraction=False))
        metadata = extractor.extract_sync(text, DocumentType.GENERIC)
        
        assert metadata.filing_date is not None
    
    def test_extract_period_end_date(self):
        text = """
        For the fiscal year ended December 31, 2024
        Period of Report: December 31, 2024
        """
        extractor = MetadataExtractor(MetadataExtractorConfig(use_llm_extraction=False))
        metadata = extractor.extract_sync(text, DocumentType.SEC_10K)
        
        assert metadata.period_end_date is not None
        assert metadata.period_end_date.year == 2024
        assert metadata.period_end_date.month == 12
    
    def test_extract_fiscal_year(self):
        text = """
        Fiscal Year End: 2024
        FY2024 Results
        """
        extractor = MetadataExtractor(MetadataExtractorConfig(use_llm_extraction=False))
        metadata = extractor.extract_sync(text, DocumentType.GENERIC)
        
        assert metadata.fiscal_year == 2024
    
    def test_extract_key_numbers(self):
        # Use text with format that matches the regex patterns
        text = """
        Total Revenue: $15.2 billion
        Net Income: $2.3 billion
        Diluted EPS: $4.52
        Total Assets: $28.5 billion
        """
        extractor = MetadataExtractor(MetadataExtractorConfig(
            use_llm_extraction=False,
            extract_key_numbers=True,
        ))
        metadata = extractor.extract_sync(text, DocumentType.SEC_10K)
        
        # Should find some financial numbers with the properly formatted text
        assert len(metadata.key_numbers) > 0
        assert "revenue" in metadata.key_numbers
    
    def test_extract_revenue(self):
        text = """
        Total Revenue: $15.2 billion
        Net Revenue: $10 million
        """
        extractor = MetadataExtractor(MetadataExtractorConfig(use_llm_extraction=False))
        metadata = extractor.extract_sync(text, DocumentType.GENERIC)
        
        assert "revenue" in metadata.key_numbers
    
    def test_extract_from_filename(self):
        text = "Generic content"
        extractor = MetadataExtractor(MetadataExtractorConfig(use_llm_extraction=False))
        metadata = extractor.extract_sync(text, DocumentType.GENERIC, doc_name="AAPL_10K_2024.txt")
        
        # Should extract ticker from filename
        assert metadata.ticker == "AAPL"
        # Should extract year from filename
        assert metadata.fiscal_year == 2024
    
    def test_extract_cik_formats(self):
        # Test 10-digit CIK
        text1 = "CIK: 0001234567"
        # Test 7-digit CIK
        text2 = "Central Index Key: 1234567"
        
        extractor = MetadataExtractor(MetadataExtractorConfig(use_llm_extraction=False))
        
        meta1 = extractor.extract_sync(text1, DocumentType.GENERIC)
        assert meta1.cik == "0001234567"
        
        meta2 = extractor.extract_sync(text2, DocumentType.GENERIC)
        assert meta2.cik == "1234567"
    
    @pytest.mark.asyncio
    async def test_fiscal_quarter_inference(self):
        """Test fiscal quarter inference - only available in async extract method"""
        text = "Period of Report: March 31, 2024"
        extractor = MetadataExtractor(MetadataExtractorConfig(use_llm_extraction=False))
        
        # Use async extract which has fiscal quarter inference
        metadata = await extractor.extract(text, DocumentType.SEC_10Q)
        
        if metadata.period_end_date:
            assert metadata.fiscal_quarter == 1  # Q1 ends in March
    
    def test_empty_text(self):
        extractor = MetadataExtractor(MetadataExtractorConfig(use_llm_extraction=False))
        metadata = extractor.extract_sync("", DocumentType.GENERIC)
        
        assert metadata.company_name is None
        assert metadata.ticker is None


class TestMetadataExtractorConfig:
    """Tests for MetadataExtractorConfig"""
    
    def test_default_config(self):
        config = MetadataExtractorConfig()
        assert config.extract_key_numbers is True
        assert config.extract_key_people is True
        assert config.use_llm_extraction is True
    
    def test_custom_config(self):
        config = MetadataExtractorConfig(
            extract_key_numbers=False,
            use_llm_extraction=False,
            sample_size=10000,
        )
        assert config.extract_key_numbers is False
        assert config.sample_size == 10000
