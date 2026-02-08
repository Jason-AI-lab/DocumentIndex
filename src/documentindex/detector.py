"""
Financial document type detection.

Detects document types:
- SEC filings (10-K, 10-Q, 8-K, DEF 14A, S-1, 20-F, 6-K)
- Earnings calls
- Research reports
- Press releases
- Generic documents
"""

import re
from typing import Optional
from .models import DocumentType


class FinancialDocDetector:
    """
    Detects financial document types from text content.
    
    Uses pattern matching on:
    - Document headers and titles
    - Filing type indicators
    - Content patterns specific to each type
    """
    
    # SEC filing patterns
    SEC_PATTERNS = {
        DocumentType.SEC_10K: [
            r'FORM\s+10-K',
            r'ANNUAL\s+REPORT\s+PURSUANT\s+TO\s+SECTION\s+13',
            r'ANNUAL\s+REPORT\s+ON\s+FORM\s+10-K',
            r'For\s+the\s+fiscal\s+year\s+ended',
        ],
        DocumentType.SEC_10Q: [
            r'FORM\s+10-Q',
            r'QUARTERLY\s+REPORT\s+PURSUANT\s+TO\s+SECTION\s+13',
            r'QUARTERLY\s+REPORT\s+ON\s+FORM\s+10-Q',
            r'For\s+the\s+(?:fiscal\s+)?quarter(?:ly\s+period)?\s+ended',
        ],
        DocumentType.SEC_8K: [
            r'FORM\s+8-K',
            r'CURRENT\s+REPORT\s+PURSUANT\s+TO\s+SECTION\s+13',
            r'CURRENT\s+REPORT\s+ON\s+FORM\s+8-K',
        ],
        DocumentType.SEC_DEF14A: [
            r'DEF\s*14A',
            r'SCHEDULE\s+14A',
            r'PROXY\s+STATEMENT',
            r'NOTICE\s+OF\s+ANNUAL\s+MEETING',
        ],
        DocumentType.SEC_S1: [
            r'FORM\s+S-1',
            r'REGISTRATION\s+STATEMENT',
            r'PROSPECTUS',
        ],
        DocumentType.SEC_20F: [
            r'FORM\s+20-F',
            r'ANNUAL\s+REPORT\s+PURSUANT\s+TO\s+SECTION\s+12',
        ],
        DocumentType.SEC_6K: [
            r'FORM\s+6-K',
            r'REPORT\s+OF\s+FOREIGN\s+PRIVATE\s+ISSUER',
        ],
    }
    
    # Earnings call patterns
    EARNINGS_CALL_PATTERNS = [
        r'earnings\s+(?:conference\s+)?call',
        r'(?:Q[1-4]|first|second|third|fourth)\s+(?:quarter\s+)?(?:\d{4}\s+)?(?:earnings|results)\s+call',
        r'operator(?:.*?)(?:good\s+(?:morning|afternoon|evening))',
        r'(?:ceo|cfo|chief\s+(?:executive|financial)\s+officer)\s+(?:will\s+)?(?:now\s+)?(?:begin|start|present)',
        r'(?:questions?\s+and\s+answers?|q\s*&\s*a)\s+session',
        r'thank\s+you\s+for\s+(?:joining|standing\s+by)',
        r'(?:i\'d\s+like\s+to|let\s+me)\s+turn\s+(?:the\s+call|it)\s+over\s+to',
    ]
    
    # Earnings release patterns
    EARNINGS_RELEASE_PATTERNS = [
        r'(?:reports?|announces?)\s+(?:Q[1-4]|first|second|third|fourth)\s+quarter\s+(?:\d{4}\s+)?(?:results|earnings)',
        r'(?:quarterly|annual)\s+(?:earnings|results)\s+(?:report|release|announcement)',
        r'press\s+release.*(?:earnings|results|revenue)',
        r'(?:gaap|non-gaap)\s+(?:eps|earnings\s+per\s+share)',
    ]
    
    # Research report patterns
    RESEARCH_REPORT_PATTERNS = [
        r'(?:buy|sell|hold|overweight|underweight|neutral)\s+rating',
        r'price\s+target\s*[:,]?\s*\$?\d+',
        r'(?:analyst|equity\s+research)\s+report',
        r'(?:initiating|maintaining|upgrading|downgrading)\s+coverage',
        r'(?:investment|analyst)\s+(?:thesis|summary|opinion)',
        r'(?:target|fair)\s+(?:price|value)\s*[:,]?\s*\$?\d+',
    ]
    
    # Press release patterns
    PRESS_RELEASE_PATTERNS = [
        r'for\s+immediate\s+release',
        r'press\s+release',
        r'news\s+release',
        r'(?:contact|media)\s*:\s*\S+@\S+',
        r'###\s*$',  # Common press release ending
    ]
    
    @classmethod
    def detect(
        cls,
        text: str,
        filename: Optional[str] = None,
    ) -> DocumentType:
        """
        Detect document type from text content and optional filename.
        
        Args:
            text: Document text content
            filename: Optional filename for additional hints
        
        Returns:
            Detected DocumentType
        """
        if not text:
            return DocumentType.GENERIC
        
        # Use first ~20K chars for detection (header area)
        sample = text[:20000].lower()
        
        # Check filename first for hints
        if filename:
            filename_lower = filename.lower()
            
            # SEC filings often have form type in filename
            if '10-k' in filename_lower or '10k' in filename_lower:
                return DocumentType.SEC_10K
            if '10-q' in filename_lower or '10q' in filename_lower:
                return DocumentType.SEC_10Q
            if '8-k' in filename_lower or '8k' in filename_lower:
                return DocumentType.SEC_8K
            if 'def14a' in filename_lower or 'proxy' in filename_lower:
                return DocumentType.SEC_DEF14A
            if 's-1' in filename_lower or 's1' in filename_lower:
                return DocumentType.SEC_S1
            if '20-f' in filename_lower or '20f' in filename_lower:
                return DocumentType.SEC_20F
            if '6-k' in filename_lower or '6k' in filename_lower:
                return DocumentType.SEC_6K
            if 'earnings' in filename_lower and 'call' in filename_lower:
                return DocumentType.EARNINGS_CALL
            if 'transcript' in filename_lower:
                return DocumentType.EARNINGS_CALL
        
        # Check SEC filing patterns
        for doc_type, patterns in cls.SEC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, sample, re.IGNORECASE):
                    return doc_type
        
        # Check earnings call patterns
        call_matches = sum(
            1 for pattern in cls.EARNINGS_CALL_PATTERNS
            if re.search(pattern, sample, re.IGNORECASE)
        )
        if call_matches >= 2:
            return DocumentType.EARNINGS_CALL
        
        # Check earnings release patterns
        release_matches = sum(
            1 for pattern in cls.EARNINGS_RELEASE_PATTERNS
            if re.search(pattern, sample, re.IGNORECASE)
        )
        if release_matches >= 2:
            return DocumentType.EARNINGS_RELEASE
        
        # Check research report patterns
        research_matches = sum(
            1 for pattern in cls.RESEARCH_REPORT_PATTERNS
            if re.search(pattern, sample, re.IGNORECASE)
        )
        if research_matches >= 2:
            return DocumentType.RESEARCH_REPORT
        
        # Check press release patterns
        press_matches = sum(
            1 for pattern in cls.PRESS_RELEASE_PATTERNS
            if re.search(pattern, sample, re.IGNORECASE)
        )
        if press_matches >= 2:
            return DocumentType.PRESS_RELEASE
        
        # Default to generic
        return DocumentType.GENERIC
    
    @classmethod
    def detect_with_confidence(
        cls,
        text: str,
        filename: Optional[str] = None,
    ) -> tuple[DocumentType, float]:
        """
        Detect document type with confidence score.
        
        Returns:
            Tuple of (DocumentType, confidence) where confidence is 0.0-1.0
        """
        if not text:
            return DocumentType.GENERIC, 0.0
        
        sample = text[:20000].lower()
        scores: dict[DocumentType, float] = {}
        
        # Score SEC filings
        for doc_type, patterns in cls.SEC_PATTERNS.items():
            matches = sum(
                1 for pattern in patterns
                if re.search(pattern, sample, re.IGNORECASE)
            )
            if matches > 0:
                scores[doc_type] = min(1.0, matches / 2)
        
        # Score earnings calls
        call_matches = sum(
            1 for pattern in cls.EARNINGS_CALL_PATTERNS
            if re.search(pattern, sample, re.IGNORECASE)
        )
        if call_matches > 0:
            scores[DocumentType.EARNINGS_CALL] = min(1.0, call_matches / 3)
        
        # Score earnings releases
        release_matches = sum(
            1 for pattern in cls.EARNINGS_RELEASE_PATTERNS
            if re.search(pattern, sample, re.IGNORECASE)
        )
        if release_matches > 0:
            scores[DocumentType.EARNINGS_RELEASE] = min(1.0, release_matches / 2)
        
        # Score research reports
        research_matches = sum(
            1 for pattern in cls.RESEARCH_REPORT_PATTERNS
            if re.search(pattern, sample, re.IGNORECASE)
        )
        if research_matches > 0:
            scores[DocumentType.RESEARCH_REPORT] = min(1.0, research_matches / 3)
        
        # Score press releases
        press_matches = sum(
            1 for pattern in cls.PRESS_RELEASE_PATTERNS
            if re.search(pattern, sample, re.IGNORECASE)
        )
        if press_matches > 0:
            scores[DocumentType.PRESS_RELEASE] = min(1.0, press_matches / 2)
        
        # Return highest scoring type
        if scores:
            best_type = max(scores, key=scores.get)  # type: ignore
            return best_type, scores[best_type]
        
        return DocumentType.GENERIC, 0.5
    
    @classmethod
    def get_section_patterns(cls, doc_type: DocumentType) -> list[tuple[str, int]]:
        """
        Get section header patterns for a document type.
        
        Returns:
            List of (pattern, level) tuples for section detection
        """
        common_patterns = [
            (r'^#{1,6}\s+', 1),  # Markdown headers
        ]
        
        if doc_type in (DocumentType.SEC_10K, DocumentType.SEC_10Q):
            return [
                (r'^PART\s+([IVX]+)\s*$', 1),
                (r'^ITEM\s+(\d+[A-Z]?)\.?\s+', 2),
                (r'^NOTE\s+(\d+)\.?\s*', 3),
                *common_patterns,
            ]
        
        elif doc_type == DocumentType.SEC_8K:
            return [
                (r'^ITEM\s+(\d+\.\d+)\s+', 1),
                (r'^SIGNATURE', 1),
                *common_patterns,
            ]
        
        elif doc_type == DocumentType.SEC_DEF14A:
            return [
                (r'^PROPOSAL\s+(\d+)', 1),
                (r'^EXECUTIVE\s+COMPENSATION', 1),
                (r'^DIRECTOR\s+COMPENSATION', 1),
                (r'^SECURITY\s+OWNERSHIP', 1),
                *common_patterns,
            ]
        
        elif doc_type == DocumentType.EARNINGS_CALL:
            return [
                (r'^(?:OPERATOR|Operator)\s*$', 1),
                (r'^(?:PRESENTATION|Presentation)\s*$', 1),
                (r'^(?:Q\s*&\s*A|QUESTION\s+AND\s+ANSWER)', 1),
                *common_patterns,
            ]
        
        elif doc_type == DocumentType.RESEARCH_REPORT:
            return [
                (r'^(?:INVESTMENT\s+)?THESIS', 1),
                (r'^(?:KEY\s+)?RISKS?', 1),
                (r'^VALUATION', 1),
                (r'^FINANCIAL\s+(?:SUMMARY|ANALYSIS)', 1),
                *common_patterns,
            ]
        
        return common_patterns
