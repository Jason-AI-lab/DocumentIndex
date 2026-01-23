"""
Metadata extraction from financial documents.

Extracts:
- Company information (name, ticker, CIK)
- Filing dates and periods
- Key financial numbers
- Important entities (people, organizations)
- Key dates mentioned
"""

from dataclasses import dataclass
from typing import Optional, Any
from datetime import datetime
import re
import logging

from .models import DocumentMetadata, DocumentType
from .llm_client import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class MetadataExtractorConfig:
    """Configuration for metadata extraction"""
    extract_key_numbers: bool = True
    extract_key_people: bool = True
    extract_key_dates: bool = True
    use_llm_extraction: bool = True  # Use LLM for complex extraction
    llm_config: Optional[LLMConfig] = None
    sample_size: int = 20000  # Characters to sample for extraction


class MetadataExtractor:
    """Extracts metadata from financial documents"""
    
    # Regex patterns for common financial document metadata
    PATTERNS = {
        # Company and Filing Info
        "cik": r"(?:CIK|Central Index Key)[:\s#]*(\d{10}|\d{7})",
        "ticker": r"(?:Trading Symbol|Ticker Symbol|Stock Symbol)[:\s]*([A-Z]{1,5})",
        "company_name": r"(?:Company Name|Registrant|REGISTRANT)[:\s]*([A-Z][A-Za-z0-9\s,\.&]+?)(?:\n|$|FORM)",
        
        # Dates
        "filing_date": r"(?:Filed|Filing Date|Date Filed)[:\s]*(\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})",
        "period_end": r"(?:For the (?:fiscal |)(?:year|quarter|period) ended?|Period of Report)[:\s]*(\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
        "fiscal_year": r"(?:Fiscal Year|FY)\s*(?:End(?:ing|ed)?)?[:\s]*(\d{4})",
        
        # Financial Numbers (with various formats)
        "revenue": r"(?:Total\s+)?(?:Net\s+)?Revenue[s]?[:\s]*\$?\s*([\d,\.]+)\s*(?:million|billion|M|B)?",
        "net_income": r"Net\s+Income[:\s]*\$?\s*([\d,\.]+)\s*(?:million|billion|M|B)?",
        "eps": r"(?:Diluted\s+)?(?:EPS|Earnings\s+[Pp]er\s+[Ss]hare)[:\s]*\$?\s*([\d\.]+)",
        "total_assets": r"Total\s+Assets[:\s]*\$?\s*([\d,\.]+)\s*(?:million|billion|M|B)?",
        "total_liabilities": r"Total\s+Liabilities[:\s]*\$?\s*([\d,\.]+)\s*(?:million|billion|M|B)?",
        "shareholders_equity": r"(?:Total\s+)?(?:Shareholders?\'?|Stockholders?\'?)\s+Equity[:\s]*\$?\s*([\d,\.]+)\s*(?:million|billion|M|B)?",
    }
    
    # Date formats to try
    DATE_FORMATS = [
        "%B %d, %Y",      # January 15, 2024
        "%B %d %Y",       # January 15 2024
        "%b %d, %Y",      # Jan 15, 2024
        "%b %d %Y",       # Jan 15 2024
        "%Y-%m-%d",       # 2024-01-15
        "%m/%d/%Y",       # 01/15/2024
        "%m-%d-%Y",       # 01-15-2024
        "%d-%b-%Y",       # 15-Jan-2024
        "%d %B %Y",       # 15 January 2024
    ]
    
    def __init__(self, config: Optional[MetadataExtractorConfig] = None):
        self.config = config or MetadataExtractorConfig()
        self.llm: Optional[LLMClient] = None
        if self.config.use_llm_extraction:
            self.llm = LLMClient(self.config.llm_config or LLMConfig())
    
    async def extract(
        self,
        text: str,
        doc_type: DocumentType,
        doc_name: Optional[str] = None,
    ) -> DocumentMetadata:
        """
        Extract metadata from document text.
        
        Args:
            text: Full document text
            doc_type: Document type
            doc_name: Optional document name for hints
        
        Returns:
            DocumentMetadata with extracted information
        """
        metadata = DocumentMetadata()
        
        # Use sample for extraction (usually header area)
        sample_text = text[:self.config.sample_size]
        
        # Extract using regex patterns (fast, no LLM needed)
        metadata.cik = self._extract_pattern("cik", sample_text)
        metadata.ticker = self._extract_pattern("ticker", sample_text)
        
        # Company name extraction
        company_name = self._extract_pattern("company_name", sample_text)
        if company_name:
            # Clean up company name
            company_name = company_name.strip().rstrip(',.')
            metadata.company_name = company_name
        
        # Date extraction
        filing_date_str = self._extract_pattern("filing_date", sample_text)
        if filing_date_str:
            metadata.filing_date = self._parse_date(filing_date_str)
        
        period_end_str = self._extract_pattern("period_end", sample_text)
        if period_end_str:
            metadata.period_end_date = self._parse_date(period_end_str)
        
        fiscal_year_str = self._extract_pattern("fiscal_year", sample_text)
        if fiscal_year_str:
            try:
                metadata.fiscal_year = int(fiscal_year_str)
            except ValueError:
                pass
        
        # Infer fiscal quarter from period end date
        if metadata.period_end_date:
            month = metadata.period_end_date.month
            if month in (1, 2, 3):
                metadata.fiscal_quarter = 1
            elif month in (4, 5, 6):
                metadata.fiscal_quarter = 2
            elif month in (7, 8, 9):
                metadata.fiscal_quarter = 3
            else:
                metadata.fiscal_quarter = 4
        
        # Key numbers extraction
        if self.config.extract_key_numbers:
            metadata.key_numbers = self._extract_key_numbers(sample_text)
        
        # Use LLM for more complex extraction if enabled
        if self.config.use_llm_extraction and self.llm:
            try:
                llm_metadata = await self._extract_with_llm(sample_text, doc_type)
                metadata = self._merge_metadata(metadata, llm_metadata)
            except Exception as e:
                logger.warning(f"LLM metadata extraction failed: {e}")
        
        # Try to extract from document name
        if doc_name:
            self._extract_from_filename(doc_name, metadata)
        
        return metadata
    
    def extract_sync(
        self,
        text: str,
        doc_type: DocumentType,
        doc_name: Optional[str] = None,
    ) -> DocumentMetadata:
        """
        Extract metadata synchronously (regex only, no LLM).
        
        Args:
            text: Full document text
            doc_type: Document type
            doc_name: Optional document name for hints
        
        Returns:
            DocumentMetadata with extracted information
        """
        metadata = DocumentMetadata()
        sample_text = text[:self.config.sample_size]
        
        # Basic extraction
        metadata.cik = self._extract_pattern("cik", sample_text)
        metadata.ticker = self._extract_pattern("ticker", sample_text)
        
        company_name = self._extract_pattern("company_name", sample_text)
        if company_name:
            metadata.company_name = company_name.strip().rstrip(',.')
        
        # Dates
        filing_date_str = self._extract_pattern("filing_date", sample_text)
        if filing_date_str:
            metadata.filing_date = self._parse_date(filing_date_str)
        
        period_end_str = self._extract_pattern("period_end", sample_text)
        if period_end_str:
            metadata.period_end_date = self._parse_date(period_end_str)
        
        fiscal_year_str = self._extract_pattern("fiscal_year", sample_text)
        if fiscal_year_str:
            try:
                metadata.fiscal_year = int(fiscal_year_str)
            except ValueError:
                pass
        
        # Key numbers
        if self.config.extract_key_numbers:
            metadata.key_numbers = self._extract_key_numbers(sample_text)
        
        # From filename
        if doc_name:
            self._extract_from_filename(doc_name, metadata)
        
        return metadata
    
    def _extract_pattern(self, pattern_name: str, text: str) -> Optional[str]:
        """Extract using regex pattern"""
        pattern = self.PATTERNS.get(pattern_name)
        if not pattern:
            return None
        
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_key_numbers(self, text: str) -> dict[str, Any]:
        """Extract key financial numbers"""
        numbers = {}
        
        for key in ["revenue", "net_income", "eps", "total_assets", 
                    "total_liabilities", "shareholders_equity"]:
            value = self._extract_pattern(key, text)
            if value:
                numbers[key] = value
        
        return numbers
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime"""
        if not date_str:
            return None
        
        date_str = date_str.strip()
        
        for fmt in self.DATE_FORMATS:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _extract_from_filename(self, filename: str, metadata: DocumentMetadata):
        """Extract metadata hints from filename"""
        # Try to extract ticker from filename (e.g., "AAPL_10K_2024.txt")
        if not metadata.ticker:
            ticker_match = re.search(r'^([A-Z]{1,5})[-_]', filename)
            if ticker_match:
                metadata.ticker = ticker_match.group(1)
        
        # Try to extract year from filename
        if not metadata.fiscal_year:
            year_match = re.search(r'[_-](\d{4})[_.-]', filename)
            if year_match:
                try:
                    metadata.fiscal_year = int(year_match.group(1))
                except ValueError:
                    pass
    
    async def _extract_with_llm(
        self,
        text: str,
        doc_type: DocumentType,
    ) -> DocumentMetadata:
        """Use LLM to extract complex metadata"""
        if not self.llm:
            return DocumentMetadata()
        
        # Limit text for prompt
        prompt_text = text[:8000]
        
        prompt = f"""Extract metadata from this {doc_type.value} financial document.

Document excerpt (beginning):
{prompt_text}

Extract and return as JSON:
{{
    "company_name": "Full legal company name or null",
    "ticker": "Stock ticker symbol or null",
    "cik": "CIK number or null",
    "filing_date": "YYYY-MM-DD format or null",
    "period_end_date": "YYYY-MM-DD format or null", 
    "fiscal_year": 2024 or null,
    "fiscal_quarter": 1-4 or null,
    "key_people": ["CEO Name", "CFO Name"],
    "key_numbers": {{
        "revenue": "$X billion or million",
        "net_income": "$X billion or million",
        "eps": "$X.XX"
    }}
}}

Only include fields you can confidently extract from the document. Use null for uncertain fields."""

        try:
            result = await self.llm.complete_json(prompt)
            
            metadata = DocumentMetadata()
            metadata.company_name = result.get("company_name")
            metadata.ticker = result.get("ticker")
            metadata.cik = result.get("cik")
            
            if result.get("filing_date"):
                metadata.filing_date = self._parse_date(result["filing_date"])
            if result.get("period_end_date"):
                metadata.period_end_date = self._parse_date(result["period_end_date"])
            
            metadata.fiscal_year = result.get("fiscal_year")
            metadata.fiscal_quarter = result.get("fiscal_quarter")
            metadata.key_people = result.get("key_people", [])
            metadata.key_numbers = result.get("key_numbers", {})
            
            return metadata
        except Exception as e:
            logger.warning(f"LLM extraction error: {e}")
            return DocumentMetadata()
    
    def _merge_metadata(
        self,
        base: DocumentMetadata,
        override: DocumentMetadata,
    ) -> DocumentMetadata:
        """Merge two metadata objects, preferring non-None values from override"""
        # Merge scalar fields
        for field_name in ["company_name", "ticker", "cik", "filing_date", 
                          "period_end_date", "fiscal_year", "fiscal_quarter"]:
            override_value = getattr(override, field_name)
            base_value = getattr(base, field_name)
            # Prefer override if base is None, or if override is not None
            if base_value is None and override_value is not None:
                setattr(base, field_name, override_value)
        
        # Merge lists (union)
        if override.key_people:
            existing = set(base.key_people)
            for person in override.key_people:
                if person and person not in existing:
                    base.key_people.append(person)
        
        # Merge dicts (override values)
        if override.key_numbers:
            for key, value in override.key_numbers.items():
                if value and key not in base.key_numbers:
                    base.key_numbers[key] = value
        
        if override.key_dates:
            existing = {d[0] for d in base.key_dates}
            for date_tuple in override.key_dates:
                if date_tuple[0] not in existing:
                    base.key_dates.append(date_tuple)
        
        return base
