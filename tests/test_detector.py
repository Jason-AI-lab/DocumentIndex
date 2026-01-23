"""
Tests for document type detection.
"""

import pytest

from documentindex.detector import FinancialDocDetector
from documentindex.models import DocumentType


class TestFinancialDocDetector:
    """Tests for document type detection"""
    
    def test_detect_10k(self, sample_10k_text):
        doc_type = FinancialDocDetector.detect(sample_10k_text)
        assert doc_type == DocumentType.SEC_10K
    
    def test_detect_10k_from_content(self):
        text = """
        FORM 10-K
        ANNUAL REPORT PURSUANT TO SECTION 13
        For the fiscal year ended December 31, 2024
        """
        doc_type = FinancialDocDetector.detect(text)
        assert doc_type == DocumentType.SEC_10K
    
    def test_detect_10q(self):
        text = """
        FORM 10-Q
        QUARTERLY REPORT PURSUANT TO SECTION 13
        For the quarterly period ended September 30, 2024
        """
        doc_type = FinancialDocDetector.detect(text)
        assert doc_type == DocumentType.SEC_10Q
    
    def test_detect_8k(self):
        text = """
        FORM 8-K
        CURRENT REPORT PURSUANT TO SECTION 13
        Date of Report: January 15, 2024
        """
        doc_type = FinancialDocDetector.detect(text)
        assert doc_type == DocumentType.SEC_8K
    
    def test_detect_proxy(self):
        text = """
        DEF 14A
        SCHEDULE 14A
        PROXY STATEMENT
        NOTICE OF ANNUAL MEETING OF SHAREHOLDERS
        """
        doc_type = FinancialDocDetector.detect(text)
        assert doc_type == DocumentType.SEC_DEF14A
    
    def test_detect_earnings_call(self, sample_earnings_call_text):
        doc_type = FinancialDocDetector.detect(sample_earnings_call_text)
        assert doc_type == DocumentType.EARNINGS_CALL
    
    def test_detect_earnings_call_patterns(self):
        text = """
        Q4 2024 Earnings Call
        
        Operator: Good afternoon, welcome to the earnings conference call.
        
        CEO: Thank you for joining us today. I'll now turn it over to our CFO.
        
        Question and Answer Session
        
        Analyst: Can you discuss the competitive environment?
        """
        doc_type = FinancialDocDetector.detect(text)
        assert doc_type == DocumentType.EARNINGS_CALL
    
    def test_detect_from_filename(self):
        text = "Some generic content"
        
        doc_type = FinancialDocDetector.detect(text, "AAPL_10K_2024.txt")
        assert doc_type == DocumentType.SEC_10K
        
        doc_type = FinancialDocDetector.detect(text, "msft_10q_q3_2024.html")
        assert doc_type == DocumentType.SEC_10Q
        
        doc_type = FinancialDocDetector.detect(text, "earnings_call_transcript.txt")
        assert doc_type == DocumentType.EARNINGS_CALL
    
    def test_detect_press_release(self):
        text = """
        FOR IMMEDIATE RELEASE
        
        ACME Corporation Reports Record Revenue
        
        Contact: media@acme.com
        
        ###
        """
        doc_type = FinancialDocDetector.detect(text)
        assert doc_type == DocumentType.PRESS_RELEASE
    
    def test_detect_research_report(self):
        text = """
        EQUITY RESEARCH REPORT
        
        Rating: BUY
        Price Target: $150
        
        Investment Thesis
        
        Initiating coverage with an overweight rating.
        """
        doc_type = FinancialDocDetector.detect(text)
        assert doc_type == DocumentType.RESEARCH_REPORT
    
    def test_detect_generic(self):
        text = "This is just some generic text without any financial document patterns."
        doc_type = FinancialDocDetector.detect(text)
        assert doc_type == DocumentType.GENERIC
    
    def test_detect_empty(self):
        doc_type = FinancialDocDetector.detect("")
        assert doc_type == DocumentType.GENERIC
    
    def test_detect_with_confidence(self, sample_10k_text):
        doc_type, confidence = FinancialDocDetector.detect_with_confidence(sample_10k_text)
        assert doc_type == DocumentType.SEC_10K
        assert confidence > 0.5
    
    def test_detect_with_confidence_generic(self):
        text = "Generic content here"
        doc_type, confidence = FinancialDocDetector.detect_with_confidence(text)
        assert doc_type == DocumentType.GENERIC
        assert confidence == 0.5  # Default for generic
    
    def test_get_section_patterns(self):
        # 10-K patterns
        patterns = FinancialDocDetector.get_section_patterns(DocumentType.SEC_10K)
        assert len(patterns) > 0
        
        # Earnings call patterns
        patterns = FinancialDocDetector.get_section_patterns(DocumentType.EARNINGS_CALL)
        assert len(patterns) > 0
        
        # Generic patterns
        patterns = FinancialDocDetector.get_section_patterns(DocumentType.GENERIC)
        assert len(patterns) > 0
