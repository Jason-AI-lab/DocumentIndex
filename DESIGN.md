# DocumentIndex: Lightweight Hierarchical Tree Index for Financial Documents

## Executive Summary

**DocumentIndex** is a lightweight, text-focused alternative to PageIndex designed specifically for building hierarchical tree indexes for long financial documents. It provides two core capabilities:

1. **Indexing**: Build a hierarchical tree structure from document text with metadata extraction
2. **Retrieval**: Two distinct retrieval modes for different use cases:
   - **Agentic QA Mode**: Intelligent, iterative retrieval for question answering
   - **Provenance Extraction Mode**: Exhaustive scan to find all evidence related to a topic

## Core Use Cases

### Use Case 1: Tree Structure Index Generation

```python
from documentindex import DocumentIndexer

indexer = DocumentIndexer(model="anthropic/claude-sonnet-4-20250514")
doc_index = await indexer.index(text, doc_name="AAPL_10K_2024")

# Output: Hierarchical tree with node IDs mapped to text positions
# {
#   "doc_name": "AAPL_10K_2024",
#   "structure": [
#     {"node_id": "0001", "title": "PART I", "start_index": 0, "end_index": 5, ...},
#     ...
#   ]
# }
```

### Use Case 2: Node Search with Text Mapping

```python
from documentindex import NodeSearcher

searcher = NodeSearcher(doc_index)
nodes = await searcher.find_related_nodes("supply chain risks")

# Returns all nodes related to query with original text mapping
for node in nodes:
    print(f"Node: {node.node_id} - {node.title}")
    print(f"Text: {node.get_text()}")  # Maps back to original text
    print(f"Relevance: {node.relevance_score}")
```

### Use Case 3a: Question Answering (Agentic Retrieval)

```python
from documentindex import AgenticQA

qa = AgenticQA(doc_index)
answer = await qa.answer(
    question="What were Apple's total revenue and gross margin in 2024?",
    max_iterations=5,
)

# Returns structured answer with citations
# {
#   "answer": "Apple's total revenue was $383B with 45.9% gross margin...",
#   "citations": [
#     {"node_id": "0023", "title": "Financial Highlights", "excerpt": "..."},
#     {"node_id": "0045", "title": "MD&A", "excerpt": "..."}
#   ],
#   "reasoning_trace": [...]
# }
```

### Use Case 3b: Provenance Extraction (Exhaustive Scan)

```python
from documentindex import ProvenanceExtractor

extractor = ProvenanceExtractor(doc_index)
evidence = await extractor.extract_all(
    topic="climate change risks and environmental commitments",
    relevance_threshold=0.6,
)

# Returns ALL nodes containing relevant evidence
# {
#   "topic": "climate change risks...",
#   "evidence": [
#     {
#       "node_id": "0012", 
#       "title": "Risk Factors",
#       "relevance_score": 0.95,
#       "excerpts": ["Climate change may...", "Environmental regulations..."],
#       "text_range": {"start": 15420, "end": 18930}
#     },
#     ...
#   ],
#   "total_nodes_scanned": 47,
#   "relevant_nodes_found": 8
# }
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DocumentIndex Package                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         INDEXING LAYER                               │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐   │    │
│  │  │ TextChunker│  │  Indexer   │  │ TreeBuilder│  │  Metadata    │   │    │
│  │  │            │─▶│            │─▶│            │─▶│  Extractor   │   │    │
│  │  └────────────┘  └────────────┘  └────────────┘  └──────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      DOCUMENT INDEX (Core Data)                      │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │    │
│  │  │  Tree        │  │  Chunk       │  │  Cross-Reference         │   │    │
│  │  │  Structure   │  │  Store       │  │  Map                     │   │    │
│  │  └──────────────┘  └──────────────┘  └──────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│                    ┌───────────────┼───────────────┐                        │
│                    ▼               ▼               ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        RETRIEVAL LAYER                               │    │
│  │                                                                       │    │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │    │
│  │  │   NodeSearcher   │  │   AgenticQA      │  │  Provenance      │   │    │
│  │  │   (find nodes)   │  │   (QA mode)      │  │  Extractor       │   │    │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘   │    │
│  │           │                     │                     │              │    │
│  │           └─────────────────────┼─────────────────────┘              │    │
│  │                                 ▼                                    │    │
│  │                    ┌──────────────────────┐                          │    │
│  │                    │  Cross-Reference     │                          │    │
│  │                    │  Resolver            │                          │    │
│  │                    └──────────────────────┘                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       INFRASTRUCTURE LAYER                           │    │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐   │    │
│  │  │ LLMClient  │  │   Cache    │  │  DocType   │  │   Utilities  │   │    │
│  │  │ (litellm)  │  │  Manager   │  │  Detector  │  │              │   │    │
│  │  └────────────┘  └────────────┘  └────────────┘  └──────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Component Design

### 1. Data Models (`models.py`)

```python
from dataclasses import dataclass, field
from typing import Optional, Any
from enum import Enum
from datetime import datetime

# ============================================================================
# Document Types
# ============================================================================

class DocumentType(Enum):
    """Financial document types with specific parsing heuristics"""
    SEC_10K = "10-K"
    SEC_10Q = "10-Q"
    SEC_8K = "8-K"
    SEC_DEF14A = "DEF 14A"
    SEC_S1 = "S-1"
    SEC_20F = "20-F"
    SEC_6K = "6-K"
    EARNINGS_CALL = "earnings_call"
    EARNINGS_RELEASE = "earnings_release"
    RESEARCH_REPORT = "research_report"
    FINANCIAL_NEWS = "financial_news"
    PRESS_RELEASE = "press_release"
    GENERIC = "generic"


# ============================================================================
# Metadata Models
# ============================================================================

@dataclass
class DocumentMetadata:
    """Extracted metadata from document"""
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    cik: Optional[str] = None
    filing_date: Optional[datetime] = None
    period_end_date: Optional[datetime] = None
    fiscal_year: Optional[int] = None
    fiscal_quarter: Optional[int] = None
    
    # Extracted entities
    key_people: list[str] = field(default_factory=list)
    key_numbers: dict[str, Any] = field(default_factory=dict)  # {"revenue": "$383B", ...}
    key_dates: list[tuple[str, datetime]] = field(default_factory=list)
    
    # Custom metadata
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class CrossReference:
    """A reference from one part of document to another"""
    source_node_id: str
    target_description: str  # e.g., "Appendix G", "Note 15", "Table 5.3"
    target_node_id: Optional[str] = None  # Resolved target
    reference_text: str = ""  # Original text containing the reference
    resolved: bool = False


# ============================================================================
# Tree Structure Models
# ============================================================================

@dataclass
class TextSpan:
    """Represents a span of text in the original document"""
    start_char: int  # Character offset in original text
    end_char: int
    start_chunk: int  # Chunk index
    end_chunk: int
    
    def __post_init__(self):
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")


@dataclass
class TreeNode:
    """A node in the hierarchical document tree"""
    node_id: str
    title: str
    level: int  # Hierarchy level (0 = root)
    
    # Text positioning - maps back to original document
    text_span: TextSpan
    
    # Content
    summary: Optional[str] = None
    
    # Hierarchy
    parent_id: Optional[str] = None
    children: list['TreeNode'] = field(default_factory=list)
    
    # Metadata specific to this node
    node_metadata: dict[str, Any] = field(default_factory=dict)
    
    # Cross-references found in this node
    cross_references: list[CrossReference] = field(default_factory=list)
    
    @property
    def start_index(self) -> int:
        """Chunk start index for backward compatibility"""
        return self.text_span.start_chunk
    
    @property
    def end_index(self) -> int:
        """Chunk end index for backward compatibility"""
        return self.text_span.end_chunk
    
    def to_dict(self, include_children: bool = True) -> dict:
        """Convert to dictionary for JSON serialization"""
        result = {
            "node_id": self.node_id,
            "title": self.title,
            "level": self.level,
            "start_index": self.text_span.start_chunk,
            "end_index": self.text_span.end_chunk,
            "start_char": self.text_span.start_char,
            "end_char": self.text_span.end_char,
        }
        if self.summary:
            result["summary"] = self.summary
        if self.parent_id:
            result["parent_id"] = self.parent_id
        if self.node_metadata:
            result["metadata"] = self.node_metadata
        if self.cross_references:
            result["cross_references"] = [
                {"target": cr.target_description, "resolved_to": cr.target_node_id}
                for cr in self.cross_references
            ]
        if include_children and self.children:
            result["nodes"] = [child.to_dict() for child in self.children]
        return result


# ============================================================================
# Document Index (Main Container)
# ============================================================================

@dataclass
class DocumentIndex:
    """Complete document index with tree structure and text mapping"""
    
    # Identity
    doc_id: str
    doc_name: str
    doc_type: DocumentType
    
    # Content
    original_text: str  # Full original text
    chunks: list[str]  # Chunked text
    chunk_char_offsets: list[tuple[int, int]]  # (start, end) char offsets for each chunk
    
    # Structure
    structure: list[TreeNode]  # Root-level nodes (full tree)
    
    # Metadata & References
    metadata: DocumentMetadata = field(default_factory=DocumentMetadata)
    cross_references: list[CrossReference] = field(default_factory=list)
    
    # Index metadata
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    index_version: str = "1.0"
    
    # -------------------------------------------------------------------------
    # Text Retrieval Methods
    # -------------------------------------------------------------------------
    
    def get_node_text(self, node_id: str) -> Optional[str]:
        """Get full text content for a specific node"""
        node = self.find_node(node_id)
        if node:
            return self.original_text[node.text_span.start_char:node.text_span.end_char]
        return None
    
    def get_chunk_text(self, start_chunk: int, end_chunk: int) -> str:
        """Get text for a range of chunks"""
        return "\n".join(self.chunks[start_chunk:end_chunk])
    
    def get_text_by_char_range(self, start: int, end: int) -> str:
        """Get text by character offset range"""
        return self.original_text[start:end]
    
    # -------------------------------------------------------------------------
    # Node Navigation Methods
    # -------------------------------------------------------------------------
    
    def find_node(self, node_id: str) -> Optional[TreeNode]:
        """Find a node by its ID"""
        return self._find_node_recursive(node_id, self.structure)
    
    def _find_node_recursive(self, node_id: str, nodes: list[TreeNode]) -> Optional[TreeNode]:
        for node in nodes:
            if node.node_id == node_id:
                return node
            if node.children:
                found = self._find_node_recursive(node_id, node.children)
                if found:
                    return found
        return None
    
    def get_all_nodes(self) -> list[TreeNode]:
        """Get flat list of all nodes in tree"""
        nodes = []
        self._collect_nodes(self.structure, nodes)
        return nodes
    
    def _collect_nodes(self, tree: list[TreeNode], result: list[TreeNode]):
        for node in tree:
            result.append(node)
            if node.children:
                self._collect_nodes(node.children, result)
    
    def get_leaf_nodes(self) -> list[TreeNode]:
        """Get all leaf nodes (nodes without children)"""
        return [n for n in self.get_all_nodes() if not n.children]
    
    def get_node_path(self, node_id: str) -> list[TreeNode]:
        """Get path from root to specified node"""
        path = []
        self._find_path(node_id, self.structure, path)
        return path
    
    def _find_path(self, node_id: str, nodes: list[TreeNode], path: list[TreeNode]) -> bool:
        for node in nodes:
            path.append(node)
            if node.node_id == node_id:
                return True
            if node.children and self._find_path(node_id, node.children, path):
                return True
            path.pop()
        return False
    
    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------
    
    def to_dict(self, include_text: bool = False) -> dict:
        """Convert to dictionary for JSON serialization"""
        result = {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "doc_type": self.doc_type.value,
            "description": self.description,
            "total_chunks": len(self.chunks),
            "total_chars": len(self.original_text),
            "created_at": self.created_at.isoformat(),
            "index_version": self.index_version,
            "metadata": {
                "company_name": self.metadata.company_name,
                "ticker": self.metadata.ticker,
                "filing_date": self.metadata.filing_date.isoformat() if self.metadata.filing_date else None,
                "key_numbers": self.metadata.key_numbers,
            },
            "structure": [node.to_dict() for node in self.structure],
        }
        if include_text:
            result["chunks"] = self.chunks
        return result
    
    @classmethod
    def from_dict(cls, data: dict, original_text: str = None) -> 'DocumentIndex':
        """Reconstruct from dictionary"""
        # Implementation for deserialization
        pass  # Full implementation in actual code


# ============================================================================
# Search & Retrieval Result Models
# ============================================================================

@dataclass
class NodeMatch:
    """A node that matches a search query"""
    node: TreeNode
    relevance_score: float  # 0.0 to 1.0
    match_reason: str  # Why this node matched
    matched_excerpts: list[str] = field(default_factory=list)  # Specific text that matched
    
    def get_text(self, doc_index: DocumentIndex) -> str:
        """Get full text of this node"""
        return doc_index.get_node_text(self.node.node_id)


@dataclass 
class Citation:
    """A citation to a specific part of the document"""
    node_id: str
    node_title: str
    excerpt: str  # The specific text being cited
    char_range: tuple[int, int]  # Position in original text


@dataclass
class QAResult:
    """Result from agentic question answering"""
    question: str
    answer: str
    confidence: float
    citations: list[Citation]
    reasoning_trace: list[dict]  # Steps taken to find answer
    nodes_visited: list[str]  # Node IDs visited during search


@dataclass
class ProvenanceResult:
    """Result from provenance extraction"""
    topic: str
    evidence: list[NodeMatch]
    total_nodes_scanned: int
    scan_coverage: float  # Percentage of document scanned
    summary: Optional[str] = None  # Optional summary of all evidence
```

### 2. LLM Client with Multi-Provider Support and Streaming (`llm_client.py`)

```python
"""
LLM Client supporting multiple providers:
- OpenAI (openai/gpt-4o, openai/gpt-4-turbo)
- Anthropic (anthropic/claude-sonnet-4-20250514, anthropic/claude-3-haiku)
- AWS Bedrock (bedrock/anthropic.claude-3-sonnet, bedrock/amazon.titan-text)
- Azure OpenAI (azure/gpt-4, azure/gpt-35-turbo)
- Local models via Ollama (ollama/llama2, ollama/mistral)
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Literal
from abc import ABC, abstractmethod
import asyncio
import json
import re
import hashlib


@dataclass
class LLMConfig:
    """Configuration for LLM client"""
    
    # Model specification (litellm format)
    # Examples:
    #   - "gpt-4o" or "openai/gpt-4o"
    #   - "anthropic/claude-sonnet-4-20250514"
    #   - "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
    #   - "azure/gpt-4-deployment-name"
    #   - "ollama/llama2"
    model: str = "gpt-4o"
    
    # Generation parameters
    temperature: float = 0.0
    max_tokens: int = 4096
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 120.0
    
    # Provider-specific configuration
    # For Azure: {"api_base": "https://xxx.openai.azure.com", "api_version": "2024-02-15"}
    # For Bedrock: {"aws_region_name": "us-east-1"}
    provider_config: dict[str, Any] = field(default_factory=dict)
    
    # API keys (if not using environment variables)
    api_key: Optional[str] = None


class LLMClient:
    """
    Async LLM client supporting multiple providers via litellm.
    
    Supported providers and model formats:
    
    OpenAI:
        model="gpt-4o"
        model="openai/gpt-4-turbo"
        
    Anthropic:
        model="anthropic/claude-sonnet-4-20250514"
        model="anthropic/claude-3-haiku-20240307"
        
    AWS Bedrock:
        model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
        model="bedrock/amazon.titan-text-express-v1"
        model="bedrock/meta.llama2-70b-chat-v1"
        provider_config={"aws_region_name": "us-east-1"}
        
    Azure OpenAI:
        model="azure/your-deployment-name"
        provider_config={
            "api_base": "https://your-resource.openai.azure.com",
            "api_version": "2024-02-15-preview"
        }
        
    Ollama (local):
        model="ollama/llama2"
        model="ollama/mistral"
    """
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self._litellm = None
    
    async def _get_litellm(self):
        """Lazy import litellm to avoid import errors if not installed"""
        if self._litellm is None:
            import litellm
            
            # Configure provider-specific settings
            if self.config.provider_config:
                for key, value in self.config.provider_config.items():
                    setattr(litellm, key, value)
            
            # Set API key if provided
            if self.config.api_key:
                litellm.api_key = self.config.api_key
            
            self._litellm = litellm
        return self._litellm
    
    async def complete(
        self,
        prompt: str,
        system_prompt: str = None,
        response_format: Literal["text", "json"] = "text",
    ) -> str:
        """
        Get completion from LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            response_format: "text" or "json" (adds JSON instruction if "json")
        
        Returns:
            LLM response text
        """
        litellm = await self._get_litellm()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        user_content = prompt
        if response_format == "json":
            user_content += "\n\nRespond with valid JSON only."
        
        messages.append({"role": "user", "content": user_content})
        
        # Build kwargs for litellm
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
        }
        
        # Add provider-specific config
        kwargs.update(self.config.provider_config)
        
        # Retry loop
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = await litellm.acompletion(**kwargs)
                return response.choices[0].message.content
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        raise RuntimeError(f"LLM call failed after {self.config.max_retries} retries: {last_error}")
    
    async def complete_json(
        self,
        prompt: str,
        system_prompt: str = None,
    ) -> dict:
        """Get JSON completion from LLM with automatic parsing"""
        response = await self.complete(prompt, system_prompt, response_format="json")
        return self._extract_json(response)
    
    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract and parse JSON from LLM response"""
        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            text = json_match.group(1)
        
        text = text.strip()
        
        # Clean common issues
        text = re.sub(r',\s*}', '}', text)
        text = re.sub(r',\s*]', ']', text)
        text = text.replace('None', 'null').replace('True', 'true').replace('False', 'false')
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object or array
            for start, end in [('{', '}'), ('[', ']')]:
                start_idx = text.find(start)
                end_idx = text.rfind(end)
                if start_idx != -1 and end_idx > start_idx:
                    try:
                        return json.loads(text[start_idx:end_idx + 1])
                    except json.JSONDecodeError:
                        continue
            return {}


# ============================================================================
# Convenience factory functions
# ============================================================================

def create_openai_client(model: str = "gpt-4o", api_key: str = None) -> LLMClient:
    """Create client for OpenAI"""
    return LLMClient(LLMConfig(
        model=model if "/" in model else f"openai/{model}",
        api_key=api_key,
    ))


def create_anthropic_client(model: str = "claude-sonnet-4-20250514", api_key: str = None) -> LLMClient:
    """Create client for Anthropic"""
    return LLMClient(LLMConfig(
        model=f"anthropic/{model}",
        api_key=api_key,
    ))


def create_bedrock_client(
    model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    region: str = "us-east-1",
) -> LLMClient:
    """
    Create client for AWS Bedrock.
    
    Requires AWS credentials configured via:
    - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - AWS credentials file (~/.aws/credentials)
    - IAM role (when running on AWS)
    
    Common Bedrock models:
    - anthropic.claude-3-sonnet-20240229-v1:0
    - anthropic.claude-3-haiku-20240307-v1:0
    - amazon.titan-text-express-v1
    - meta.llama2-70b-chat-v1
    """
    return LLMClient(LLMConfig(
        model=f"bedrock/{model}",
        provider_config={"aws_region_name": region},
    ))


def create_azure_client(
    deployment_name: str,
    api_base: str,
    api_version: str = "2024-02-15-preview",
    api_key: str = None,
) -> LLMClient:
    """
    Create client for Azure OpenAI.
    
    Args:
        deployment_name: Your Azure deployment name
        api_base: Your Azure endpoint (https://xxx.openai.azure.com)
        api_version: API version
        api_key: Azure API key (or use AZURE_API_KEY env var)
    """
    return LLMClient(LLMConfig(
        model=f"azure/{deployment_name}",
        api_key=api_key,
        provider_config={
            "api_base": api_base,
            "api_version": api_version,
        },
    ))
```

### 3. Cache Manager (`cache.py`)

```python
"""
Caching layer for DocumentIndex.

Supports multiple backends:
- Memory (default, for development)
- File system (for persistence)
- Redis (for distributed/production)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any, TypeVar, Generic
from datetime import datetime, timedelta
import json
import hashlib
import pickle
from pathlib import Path

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """A cached item with metadata"""
    key: str
    value: T
    created_at: datetime
    expires_at: Optional[datetime] = None
    hit_count: int = 0
    
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class CacheBackend(ABC):
    """Abstract cache backend"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set value in cache with optional TTL in seconds"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cache entries"""
        pass


class MemoryCache(CacheBackend):
    """In-memory cache (for development/testing)"""
    
    def __init__(self, max_size: int = 1000):
        self._cache: dict[str, CacheEntry] = {}
        self._max_size = max_size
    
    async def get(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._cache[key]
            return None
        entry.hit_count += 1
        return entry.value
    
    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        # Evict if at capacity (LRU-like: remove least hit)
        if len(self._cache) >= self._max_size and key not in self._cache:
            min_key = min(self._cache.keys(), key=lambda k: self._cache[k].hit_count)
            del self._cache[min_key]
        
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None
        self._cache[key] = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            expires_at=expires_at,
        )
    
    async def delete(self, key: str) -> None:
        self._cache.pop(key, None)
    
    async def exists(self, key: str) -> bool:
        entry = self._cache.get(key)
        if entry and entry.is_expired():
            del self._cache[key]
            return False
        return key in self._cache
    
    async def clear(self) -> None:
        self._cache.clear()


class FileCache(CacheBackend):
    """File-based cache for persistence"""
    
    def __init__(self, cache_dir: str = ".cache/documentindex"):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, key: str) -> Path:
        # Hash key for filesystem-safe filename
        hashed = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self._cache_dir / f"{hashed}.cache"
    
    async def get(self, key: str) -> Optional[Any]:
        path = self._get_path(key)
        if not path.exists():
            return None
        
        try:
            with open(path, 'rb') as f:
                entry: CacheEntry = pickle.load(f)
            
            if entry.is_expired():
                path.unlink()
                return None
            
            return entry.value
        except Exception:
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        expires_at = datetime.now() + timedelta(seconds=ttl) if ttl else None
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            expires_at=expires_at,
        )
        
        path = self._get_path(key)
        with open(path, 'wb') as f:
            pickle.dump(entry, f)
    
    async def delete(self, key: str) -> None:
        path = self._get_path(key)
        if path.exists():
            path.unlink()
    
    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None
    
    async def clear(self) -> None:
        for path in self._cache_dir.glob("*.cache"):
            path.unlink()


class RedisCache(CacheBackend):
    """Redis-based cache for distributed/production use"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        prefix: str = "docindex:",
        password: str = None,
    ):
        self._prefix = prefix
        self._redis = None
        self._config = {
            "host": host,
            "port": port,
            "db": db,
            "password": password,
        }
    
    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as redis
            self._redis = redis.Redis(**self._config, decode_responses=False)
        return self._redis
    
    def _make_key(self, key: str) -> str:
        return f"{self._prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        r = await self._get_redis()
        data = await r.get(self._make_key(key))
        if data is None:
            return None
        return pickle.loads(data)
    
    async def set(self, key: str, value: Any, ttl: int = None) -> None:
        r = await self._get_redis()
        data = pickle.dumps(value)
        if ttl:
            await r.setex(self._make_key(key), ttl, data)
        else:
            await r.set(self._make_key(key), data)
    
    async def delete(self, key: str) -> None:
        r = await self._get_redis()
        await r.delete(self._make_key(key))
    
    async def exists(self, key: str) -> bool:
        r = await self._get_redis()
        return await r.exists(self._make_key(key))
    
    async def clear(self) -> None:
        r = await self._get_redis()
        keys = await r.keys(f"{self._prefix}*")
        if keys:
            await r.delete(*keys)


# ============================================================================
# Cache Manager (High-level interface)
# ============================================================================

@dataclass
class CacheConfig:
    """Cache configuration"""
    backend: str = "memory"  # "memory", "file", "redis"
    
    # Memory cache settings
    memory_max_size: int = 1000
    
    # File cache settings
    file_cache_dir: str = ".cache/documentindex"
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = None
    redis_prefix: str = "docindex:"
    
    # TTL defaults (in seconds)
    index_ttl: int = 86400 * 7  # 7 days for document indexes
    llm_response_ttl: int = 3600  # 1 hour for LLM responses
    search_result_ttl: int = 1800  # 30 minutes for search results


class CacheManager:
    """
    High-level cache manager for DocumentIndex.
    
    Caches:
    - Document indexes (keyed by doc_id or content hash)
    - LLM responses (keyed by prompt hash)
    - Search results (keyed by query + doc_id)
    """
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self._backend = self._create_backend()
    
    def _create_backend(self) -> CacheBackend:
        if self.config.backend == "memory":
            return MemoryCache(self.config.memory_max_size)
        elif self.config.backend == "file":
            return FileCache(self.config.file_cache_dir)
        elif self.config.backend == "redis":
            return RedisCache(
                host=self.config.redis_host,
                port=self.config.redis_port,
                db=self.config.redis_db,
                password=self.config.redis_password,
                prefix=self.config.redis_prefix,
            )
        else:
            raise ValueError(f"Unknown cache backend: {self.config.backend}")
    
    # -------------------------------------------------------------------------
    # Key generation
    # -------------------------------------------------------------------------
    
    @staticmethod
    def _hash_content(content: str) -> str:
        """Generate hash for content"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _index_key(self, doc_id: str) -> str:
        return f"index:{doc_id}"
    
    def _llm_key(self, prompt: str, model: str) -> str:
        prompt_hash = self._hash_content(prompt)
        return f"llm:{model}:{prompt_hash}"
    
    def _search_key(self, doc_id: str, query: str) -> str:
        query_hash = self._hash_content(query)
        return f"search:{doc_id}:{query_hash}"
    
    # -------------------------------------------------------------------------
    # Document Index caching
    # -------------------------------------------------------------------------
    
    async def get_index(self, doc_id: str) -> Optional[dict]:
        """Get cached document index"""
        return await self._backend.get(self._index_key(doc_id))
    
    async def set_index(self, doc_id: str, index_data: dict) -> None:
        """Cache document index"""
        await self._backend.set(
            self._index_key(doc_id),
            index_data,
            ttl=self.config.index_ttl,
        )
    
    async def has_index(self, doc_id: str) -> bool:
        """Check if index is cached"""
        return await self._backend.exists(self._index_key(doc_id))
    
    # -------------------------------------------------------------------------
    # LLM Response caching
    # -------------------------------------------------------------------------
    
    async def get_llm_response(self, prompt: str, model: str) -> Optional[str]:
        """Get cached LLM response"""
        return await self._backend.get(self._llm_key(prompt, model))
    
    async def set_llm_response(self, prompt: str, model: str, response: str) -> None:
        """Cache LLM response"""
        await self._backend.set(
            self._llm_key(prompt, model),
            response,
            ttl=self.config.llm_response_ttl,
        )
    
    # -------------------------------------------------------------------------
    # Search Result caching
    # -------------------------------------------------------------------------
    
    async def get_search_result(self, doc_id: str, query: str) -> Optional[Any]:
        """Get cached search result"""
        return await self._backend.get(self._search_key(doc_id, query))
    
    async def set_search_result(self, doc_id: str, query: str, result: Any) -> None:
        """Cache search result"""
        await self._backend.set(
            self._search_key(doc_id, query),
            result,
            ttl=self.config.search_result_ttl,
        )
```

### 4. Metadata Extractor (`metadata.py`)

```python
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
from .models import DocumentMetadata, DocumentType
from .llm_client import LLMClient, LLMConfig


@dataclass
class MetadataExtractorConfig:
    """Configuration for metadata extraction"""
    extract_key_numbers: bool = True
    extract_key_people: bool = True
    extract_key_dates: bool = True
    use_llm_extraction: bool = True  # Use LLM for complex extraction
    llm_config: LLMConfig = None


class MetadataExtractor:
    """Extracts metadata from financial documents"""
    
    # Regex patterns for common financial document metadata
    PATTERNS = {
        # Company and Filing Info
        "cik": r"(?:CIK|Central Index Key)[:\s]*(\d{10})",
        "ticker": r"(?:Trading Symbol|Ticker Symbol)[:\s]*([A-Z]{1,5})",
        "company_name": r"(?:Company Name|Registrant)[:\s]*([A-Z][A-Za-z0-9\s,\.]+?)(?:\n|$)",
        
        # Dates
        "filing_date": r"(?:Filed|Filing Date)[:\s]*(\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
        "period_end": r"(?:For the (?:fiscal |)(?:year|quarter|period) ended?|Period of Report)[:\s]*(\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
        "fiscal_year": r"(?:Fiscal Year|FY)\s*(?:End(?:ing|ed)?)?[:\s]*(\d{4})",
        
        # Financial Numbers (with various formats)
        "revenue": r"(?:Total\s+)?(?:Net\s+)?Revenue[s]?[:\s]*\$?\s*([\d,\.]+)\s*(?:million|billion|M|B)?",
        "net_income": r"Net\s+Income[:\s]*\$?\s*([\d,\.]+)\s*(?:million|billion|M|B)?",
        "eps": r"(?:Diluted\s+)?(?:EPS|Earnings\s+[Pp]er\s+[Ss]hare)[:\s]*\$?\s*([\d\.]+)",
        "total_assets": r"Total\s+Assets[:\s]*\$?\s*([\d,\.]+)\s*(?:million|billion|M|B)?",
    }
    
    def __init__(self, config: MetadataExtractorConfig = None):
        self.config = config or MetadataExtractorConfig()
        if self.config.use_llm_extraction:
            self.llm = LLMClient(self.config.llm_config or LLMConfig())
        else:
            self.llm = None
    
    async def extract(
        self,
        text: str,
        doc_type: DocumentType,
        doc_name: str = None,
    ) -> DocumentMetadata:
        """Extract metadata from document text"""
        
        metadata = DocumentMetadata()
        
        # Extract using regex patterns (fast, no LLM needed)
        sample_text = text[:20000]  # Use first 20K chars for metadata
        
        # Basic extraction
        metadata.cik = self._extract_pattern("cik", sample_text)
        metadata.ticker = self._extract_pattern("ticker", sample_text)
        metadata.company_name = self._extract_pattern("company_name", sample_text)
        
        # Date extraction
        filing_date_str = self._extract_pattern("filing_date", sample_text)
        if filing_date_str:
            metadata.filing_date = self._parse_date(filing_date_str)
        
        period_end_str = self._extract_pattern("period_end", sample_text)
        if period_end_str:
            metadata.period_end_date = self._parse_date(period_end_str)
        
        fiscal_year_str = self._extract_pattern("fiscal_year", sample_text)
        if fiscal_year_str:
            metadata.fiscal_year = int(fiscal_year_str)
        
        # Key numbers
        if self.config.extract_key_numbers:
            metadata.key_numbers = self._extract_key_numbers(sample_text)
        
        # Use LLM for more complex extraction if enabled
        if self.config.use_llm_extraction and self.llm:
            llm_metadata = await self._extract_with_llm(sample_text, doc_type)
            metadata = self._merge_metadata(metadata, llm_metadata)
        
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
        
        for key in ["revenue", "net_income", "eps", "total_assets"]:
            value = self._extract_pattern(key, text)
            if value:
                numbers[key] = value
        
        return numbers
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime"""
        formats = [
            "%B %d, %Y",      # January 15, 2024
            "%B %d %Y",       # January 15 2024
            "%Y-%m-%d",       # 2024-01-15
            "%m/%d/%Y",       # 01/15/2024
            "%d-%b-%Y",       # 15-Jan-2024
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None
    
    async def _extract_with_llm(
        self,
        text: str,
        doc_type: DocumentType,
    ) -> DocumentMetadata:
        """Use LLM to extract complex metadata"""
        
        prompt = f"""Extract metadata from this {doc_type.value} financial document.

Document excerpt (first part):
{text[:8000]}

Extract and return as JSON:
{{
    "company_name": "Full legal company name",
    "ticker": "Stock ticker symbol",
    "filing_date": "YYYY-MM-DD format",
    "period_end_date": "YYYY-MM-DD format", 
    "fiscal_year": 2024,
    "fiscal_quarter": 1-4 or null,
    "key_people": ["CEO Name", "CFO Name"],
    "key_numbers": {{
        "revenue": "$X billion",
        "net_income": "$X million",
        "eps": "$X.XX"
    }}
}}

Only include fields you can confidently extract. Use null for uncertain fields."""

        try:
            result = await self.llm.complete_json(prompt)
            
            metadata = DocumentMetadata()
            metadata.company_name = result.get("company_name")
            metadata.ticker = result.get("ticker")
            
            if result.get("filing_date"):
                metadata.filing_date = self._parse_date(result["filing_date"])
            if result.get("period_end_date"):
                metadata.period_end_date = self._parse_date(result["period_end_date"])
            
            metadata.fiscal_year = result.get("fiscal_year")
            metadata.fiscal_quarter = result.get("fiscal_quarter")
            metadata.key_people = result.get("key_people", [])
            metadata.key_numbers = result.get("key_numbers", {})
            
            return metadata
        except Exception:
            return DocumentMetadata()
    
    def _merge_metadata(
        self,
        base: DocumentMetadata,
        override: DocumentMetadata,
    ) -> DocumentMetadata:
        """Merge two metadata objects, preferring non-None values from override"""
        for field in ["company_name", "ticker", "cik", "filing_date", 
                      "period_end_date", "fiscal_year", "fiscal_quarter"]:
            override_value = getattr(override, field)
            if override_value is not None:
                setattr(base, field, override_value)
        
        # Merge lists and dicts
        if override.key_people:
            base.key_people = list(set(base.key_people + override.key_people))
        if override.key_numbers:
            base.key_numbers.update(override.key_numbers)
        
        return base
```

### 5. Cross-Reference Resolver (`cross_ref.py`)

```python
"""
Cross-reference detection and resolution.

Detects references like:
- "See Appendix G"
- "Refer to Note 15"
- "As shown in Table 5.3"
- "Described in Item 1A"

And resolves them to actual node IDs in the document tree.
"""

from dataclasses import dataclass
from typing import Optional
import re
from .models import DocumentIndex, TreeNode, CrossReference
from .llm_client import LLMClient, LLMConfig


@dataclass
class CrossRefConfig:
    """Configuration for cross-reference resolution"""
    use_llm_resolution: bool = True  # Use LLM for ambiguous references
    llm_config: LLMConfig = None


class CrossReferenceResolver:
    """Detects and resolves cross-references in documents"""
    
    # Patterns for detecting cross-references
    REFERENCE_PATTERNS = [
        # Appendix references
        (r"(?:see|refer to|in|described in)\s+Appendix\s+([A-Z](?:\d+)?)", "appendix"),
        # Note references (financial statements)
        (r"(?:see|refer to)\s+Note\s+(\d+(?:\.\d+)?)", "note"),
        # Item references (SEC filings)
        (r"(?:see|refer to|in|described in)\s+Item\s+(\d+[A-Z]?)", "item"),
        # Table references
        (r"(?:see|in|shown in)\s+Table\s+(\d+(?:\.\d+)?)", "table"),
        # Figure references
        (r"(?:see|in|shown in)\s+Figure\s+(\d+(?:\.\d+)?)", "figure"),
        # Section references
        (r"(?:see|refer to|in)\s+Section\s+(\d+(?:\.\d+)*)", "section"),
        # Exhibit references
        (r"(?:see|refer to)\s+Exhibit\s+(\d+(?:\.\d+)?)", "exhibit"),
        # Part references
        (r"(?:see|in)\s+Part\s+([IVX]+|\d+)", "part"),
    ]
    
    def __init__(self, config: CrossRefConfig = None):
        self.config = config or CrossRefConfig()
        if self.config.use_llm_resolution:
            self.llm = LLMClient(self.config.llm_config or LLMConfig())
        else:
            self.llm = None
    
    def detect_references(self, text: str) -> list[CrossReference]:
        """Detect all cross-references in text"""
        references = []
        
        for pattern, ref_type in self.REFERENCE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                full_match = match.group(0)
                ref_target = match.group(1)
                
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                references.append(CrossReference(
                    source_node_id="",  # Will be filled by caller
                    target_description=f"{ref_type.title()} {ref_target}",
                    reference_text=context,
                    resolved=False,
                ))
        
        return references
    
    async def resolve_references(
        self,
        doc_index: DocumentIndex,
    ) -> list[CrossReference]:
        """Resolve all cross-references to their target nodes"""
        
        all_references = []
        
        # Collect all references from all nodes
        for node in doc_index.get_all_nodes():
            node_text = doc_index.get_node_text(node.node_id)
            if not node_text:
                continue
            
            refs = self.detect_references(node_text)
            for ref in refs:
                ref.source_node_id = node.node_id
            
            all_references.extend(refs)
            node.cross_references = refs
        
        # Resolve each reference
        for ref in all_references:
            target_node = self._find_target_node(ref.target_description, doc_index)
            if target_node:
                ref.target_node_id = target_node.node_id
                ref.resolved = True
        
        # Use LLM for unresolved references
        if self.config.use_llm_resolution and self.llm:
            unresolved = [r for r in all_references if not r.resolved]
            if unresolved:
                await self._resolve_with_llm(unresolved, doc_index)
        
        doc_index.cross_references = all_references
        return all_references
    
    def _find_target_node(
        self,
        target_description: str,
        doc_index: DocumentIndex,
    ) -> Optional[TreeNode]:
        """Find node matching target description"""
        
        # Normalize target for matching
        target_lower = target_description.lower()
        
        for node in doc_index.get_all_nodes():
            title_lower = node.title.lower()
            
            # Direct title match
            if target_lower in title_lower or title_lower in target_lower:
                return node
            
            # Pattern-based matching
            if self._titles_match(target_description, node.title):
                return node
        
        return None
    
    def _titles_match(self, target: str, title: str) -> bool:
        """Check if target reference matches node title"""
        target_lower = target.lower()
        title_lower = title.lower()
        
        # Extract type and number from target
        match = re.match(r"(\w+)\s+([A-Z0-9\.]+)", target, re.IGNORECASE)
        if not match:
            return False
        
        ref_type, ref_num = match.groups()
        ref_type = ref_type.lower()
        ref_num = ref_num.lower()
        
        # Check if title contains both type and number
        if ref_type in title_lower and ref_num in title_lower:
            return True
        
        # Special handling for notes: "Note 15" might be "15. Revenue Recognition"
        if ref_type == "note":
            if re.match(rf"^{ref_num}\.", title_lower):
                return True
        
        return False
    
    async def _resolve_with_llm(
        self,
        references: list[CrossReference],
        doc_index: DocumentIndex,
    ) -> None:
        """Use LLM to resolve ambiguous references"""
        
        # Build node list for LLM
        node_list = []
        for node in doc_index.get_all_nodes():
            node_list.append(f"[{node.node_id}] {node.title}")
        node_text = "\n".join(node_list)
        
        for ref in references:
            prompt = f"""Match this cross-reference to the correct document section.

Cross-reference: "{ref.target_description}"
Context: "{ref.reference_text}"

Available sections:
{node_text}

Return JSON with the matching node_id, or null if no match:
{{"node_id": "0001"}} or {{"node_id": null}}"""

            try:
                result = await self.llm.complete_json(prompt)
                if result.get("node_id"):
                    ref.target_node_id = result["node_id"]
                    ref.resolved = True
            except Exception:
                continue


class CrossReferenceFollower:
    """Utility to follow cross-references during retrieval"""
    
    def __init__(self, doc_index: DocumentIndex):
        self.doc_index = doc_index
    
    def get_referenced_nodes(self, node_id: str) -> list[TreeNode]:
        """Get all nodes referenced by a given node"""
        node = self.doc_index.find_node(node_id)
        if not node:
            return []
        
        referenced = []
        for ref in node.cross_references:
            if ref.resolved and ref.target_node_id:
                target = self.doc_index.find_node(ref.target_node_id)
                if target:
                    referenced.append(target)
        
        return referenced
    
    def get_referencing_nodes(self, node_id: str) -> list[TreeNode]:
        """Get all nodes that reference a given node"""
        referencing = []
        
        for node in self.doc_index.get_all_nodes():
            for ref in node.cross_references:
                if ref.target_node_id == node_id:
                    referencing.append(node)
                    break
        
        return referencing
```

### 6. Node Searcher (`searcher.py`)

```python
"""
Node search functionality - finds all nodes related to a query.

This is the foundation for both:
- Agentic QA (selective, iterative retrieval)
- Provenance Extraction (exhaustive scan)
"""

from dataclasses import dataclass
from typing import Optional
import asyncio
from .models import DocumentIndex, TreeNode, NodeMatch
from .llm_client import LLMClient, LLMConfig
from .cache import CacheManager


@dataclass
class NodeSearchConfig:
    """Configuration for node search"""
    relevance_threshold: float = 0.5  # Minimum relevance score
    max_results: int = 20
    include_children: bool = True  # Include children of matching nodes
    follow_cross_refs: bool = True  # Follow cross-references
    use_cache: bool = True


class NodeSearcher:
    """
    Searches document tree for nodes related to a query.
    
    Returns all matching nodes with:
    - Relevance scores
    - Match reasoning
    - Text mapping to original document
    """
    
    def __init__(
        self,
        doc_index: DocumentIndex,
        llm_config: LLMConfig = None,
        cache_manager: CacheManager = None,
    ):
        self.doc_index = doc_index
        self.llm = LLMClient(llm_config or LLMConfig())
        self.cache = cache_manager
    
    async def find_related_nodes(
        self,
        query: str,
        config: NodeSearchConfig = None,
    ) -> list[NodeMatch]:
        """
        Find all nodes related to the query.
        
        Args:
            query: Search query or topic
            config: Search configuration
        
        Returns:
            List of matching nodes with relevance scores
        """
        config = config or NodeSearchConfig()
        
        # Check cache
        if config.use_cache and self.cache:
            cached = await self.cache.get_search_result(self.doc_index.doc_id, query)
            if cached:
                return cached
        
        # Get all nodes to evaluate
        all_nodes = self.doc_index.get_all_nodes()
        
        # Score each node for relevance
        matches = await self._score_nodes(all_nodes, query, config)
        
        # Filter by threshold and limit
        matches = [m for m in matches if m.relevance_score >= config.relevance_threshold]
        matches.sort(key=lambda m: m.relevance_score, reverse=True)
        matches = matches[:config.max_results]
        
        # Follow cross-references if enabled
        if config.follow_cross_refs:
            matches = await self._expand_with_cross_refs(matches, query, config)
        
        # Cache results
        if config.use_cache and self.cache:
            await self.cache.set_search_result(self.doc_index.doc_id, query, matches)
        
        return matches
    
    async def _score_nodes(
        self,
        nodes: list[TreeNode],
        query: str,
        config: NodeSearchConfig,
    ) -> list[NodeMatch]:
        """Score all nodes for relevance to query"""
        
        # Batch nodes for efficient LLM calls
        batch_size = 10
        batches = [nodes[i:i+batch_size] for i in range(0, len(nodes), batch_size)]
        
        all_matches = []
        tasks = [self._score_batch(batch, query) for batch in batches]
        results = await asyncio.gather(*tasks)
        
        for batch_matches in results:
            all_matches.extend(batch_matches)
        
        return all_matches
    
    async def _score_batch(
        self,
        nodes: list[TreeNode],
        query: str,
    ) -> list[NodeMatch]:
        """Score a batch of nodes"""
        
        # Build node descriptions
        node_descriptions = []
        for node in nodes:
            desc = f"[{node.node_id}] {node.title}"
            if node.summary:
                desc += f" - {node.summary[:200]}"
            node_descriptions.append(desc)
        
        nodes_text = "\n".join(node_descriptions)
        
        prompt = f"""Evaluate how relevant each document section is to this query/topic.

Query: {query}

Sections:
{nodes_text}

For each section, provide a relevance score (0.0-1.0) and brief reasoning.

Return JSON array:
[
  {{"node_id": "0001", "score": 0.8, "reason": "Contains discussion of..."}},
  ...
]

Score guidelines:
- 1.0: Directly addresses the query
- 0.7-0.9: Highly relevant, contains key information
- 0.4-0.6: Somewhat relevant, may contain useful context
- 0.1-0.3: Tangentially related
- 0.0: Not relevant"""

        try:
            results = await self.llm.complete_json(prompt)
            
            matches = []
            result_map = {r["node_id"]: r for r in results if isinstance(r, dict)}
            
            for node in nodes:
                if node.node_id in result_map:
                    r = result_map[node.node_id]
                    matches.append(NodeMatch(
                        node=node,
                        relevance_score=float(r.get("score", 0)),
                        match_reason=r.get("reason", ""),
                    ))
                else:
                    # Node not in results, assume low relevance
                    matches.append(NodeMatch(
                        node=node,
                        relevance_score=0.0,
                        match_reason="Not evaluated",
                    ))
            
            return matches
        except Exception:
            # On error, return all with neutral score
            return [
                NodeMatch(node=n, relevance_score=0.5, match_reason="Evaluation error")
                for n in nodes
            ]
    
    async def _expand_with_cross_refs(
        self,
        matches: list[NodeMatch],
        query: str,
        config: NodeSearchConfig,
    ) -> list[NodeMatch]:
        """Expand results by following cross-references"""
        
        matched_ids = {m.node.node_id for m in matches}
        additional = []
        
        for match in matches:
            for ref in match.node.cross_references:
                if ref.resolved and ref.target_node_id not in matched_ids:
                    target = self.doc_index.find_node(ref.target_node_id)
                    if target:
                        # Score the referenced node
                        ref_matches = await self._score_batch([target], query)
                        if ref_matches and ref_matches[0].relevance_score >= config.relevance_threshold:
                            additional.append(ref_matches[0])
                            matched_ids.add(target.node_id)
        
        return matches + additional
```

### 7. Agentic QA (`agentic_qa.py`)

```python
"""
Agentic Question Answering - iterative, intelligent retrieval for QA.

Workflow:
1. Understand the question
2. Examine document structure
3. Select most promising section
4. Read and extract information
5. If sufficient: answer; else: continue searching
6. Follow cross-references as needed
"""

from dataclasses import dataclass, field
from typing import Optional
import asyncio
from .models import DocumentIndex, TreeNode, QAResult, Citation
from .llm_client import LLMClient, LLMConfig
from .cross_ref import CrossReferenceFollower
from .cache import CacheManager


@dataclass
class AgenticQAConfig:
    """Configuration for agentic QA"""
    max_iterations: int = 5
    max_context_tokens: int = 8000
    follow_cross_refs: bool = True
    generate_citations: bool = True
    confidence_threshold: float = 0.7  # Min confidence to stop searching


@dataclass
class ReasoningStep:
    """A step in the reasoning process"""
    step_num: int
    action: str  # "examine_structure", "read_section", "follow_reference", "answer"
    node_id: Optional[str] = None
    reasoning: str = ""
    findings: str = ""


class AgenticQA:
    """
    Agentic question answering system.
    
    Uses iterative reasoning to find and synthesize information
    from the document to answer questions.
    """
    
    def __init__(
        self,
        doc_index: DocumentIndex,
        llm_config: LLMConfig = None,
        cache_manager: CacheManager = None,
    ):
        self.doc_index = doc_index
        self.llm = LLMClient(llm_config or LLMConfig())
        self.cache = cache_manager
        self.cross_ref_follower = CrossReferenceFollower(doc_index)
    
    async def answer(
        self,
        question: str,
        config: AgenticQAConfig = None,
    ) -> QAResult:
        """
        Answer a question using agentic retrieval.
        
        Args:
            question: The question to answer
            config: QA configuration
        
        Returns:
            QAResult with answer, citations, and reasoning trace
        """
        config = config or AgenticQAConfig()
        
        # Initialize state
        reasoning_trace: list[ReasoningStep] = []
        visited_nodes: list[str] = []
        gathered_info: list[tuple[str, str, str]] = []  # (node_id, title, excerpt)
        
        # Step 1: Understand question and plan
        plan = await self._plan_search(question)
        reasoning_trace.append(ReasoningStep(
            step_num=1,
            action="plan",
            reasoning=plan["reasoning"],
            findings=f"Search targets: {plan['targets']}",
        ))
        
        # Iterative search loop
        for iteration in range(config.max_iterations):
            # Get current structure view
            structure_view = self._get_structure_view(visited_nodes)
            
            # Decide next action
            decision = await self._decide_next_action(
                question=question,
                structure=structure_view,
                gathered_info=gathered_info,
                plan=plan,
            )
            
            action = decision["action"]
            
            if action == "answer":
                # We have enough info to answer
                reasoning_trace.append(ReasoningStep(
                    step_num=len(reasoning_trace) + 1,
                    action="answer",
                    reasoning=decision["reasoning"],
                ))
                break
            
            elif action == "read_section":
                node_id = decision["node_id"]
                visited_nodes.append(node_id)
                
                # Read the section
                node = self.doc_index.find_node(node_id)
                text = self.doc_index.get_node_text(node_id)
                
                if text:
                    # Extract relevant information
                    extraction = await self._extract_relevant_info(question, text)
                    
                    if extraction["found_relevant"]:
                        gathered_info.append((
                            node_id,
                            node.title,
                            extraction["excerpt"],
                        ))
                    
                    reasoning_trace.append(ReasoningStep(
                        step_num=len(reasoning_trace) + 1,
                        action="read_section",
                        node_id=node_id,
                        reasoning=decision["reasoning"],
                        findings=extraction["summary"],
                    ))
                    
                    # Follow cross-references if enabled
                    if config.follow_cross_refs and extraction.get("follow_refs"):
                        for ref in node.cross_references:
                            if ref.resolved and ref.target_node_id not in visited_nodes:
                                reasoning_trace.append(ReasoningStep(
                                    step_num=len(reasoning_trace) + 1,
                                    action="follow_reference",
                                    node_id=ref.target_node_id,
                                    reasoning=f"Following reference to {ref.target_description}",
                                ))
            
            elif action == "give_up":
                reasoning_trace.append(ReasoningStep(
                    step_num=len(reasoning_trace) + 1,
                    action="give_up",
                    reasoning=decision["reasoning"],
                ))
                break
        
        # Generate final answer
        answer_result = await self._generate_answer(
            question=question,
            gathered_info=gathered_info,
            config=config,
        )
        
        # Build citations
        citations = []
        if config.generate_citations:
            for node_id, title, excerpt in gathered_info:
                node = self.doc_index.find_node(node_id)
                citations.append(Citation(
                    node_id=node_id,
                    node_title=title,
                    excerpt=excerpt[:500],
                    char_range=(node.text_span.start_char, node.text_span.end_char),
                ))
        
        return QAResult(
            question=question,
            answer=answer_result["answer"],
            confidence=answer_result["confidence"],
            citations=citations,
            reasoning_trace=[vars(step) for step in reasoning_trace],
            nodes_visited=visited_nodes,
        )
    
    async def _plan_search(self, question: str) -> dict:
        """Plan the search strategy"""
        
        # Get document structure summary
        structure_summary = self._get_structure_summary()
        
        prompt = f"""Plan how to find the answer to this question in the document.

Document: {self.doc_index.description}
Document Structure:
{structure_summary}

Question: {question}

Think about:
1. What type of information is needed?
2. Which sections likely contain this information?
3. Are there multiple places to check?

Return JSON:
{{
  "reasoning": "explanation of search strategy",
  "targets": ["section types or titles to look for"],
  "priority": "high/medium/low confidence this can be answered"
}}"""

        return await self.llm.complete_json(prompt)
    
    async def _decide_next_action(
        self,
        question: str,
        structure: str,
        gathered_info: list,
        plan: dict,
    ) -> dict:
        """Decide the next action to take"""
        
        info_summary = ""
        if gathered_info:
            info_summary = "Information gathered so far:\n"
            for node_id, title, excerpt in gathered_info:
                info_summary += f"- [{node_id}] {title}: {excerpt[:200]}...\n"
        
        prompt = f"""Decide the next action to answer this question.

Question: {question}

Document Structure (node_id in brackets):
{structure}

{info_summary}

What should we do next?

Options:
1. "read_section" - Read a specific section (provide node_id)
2. "answer" - We have enough information to answer
3. "give_up" - Cannot find the answer in this document

Return JSON:
{{
  "action": "read_section|answer|give_up",
  "node_id": "0001",  // only for read_section
  "reasoning": "why this action"
}}"""

        return await self.llm.complete_json(prompt)
    
    async def _extract_relevant_info(self, question: str, text: str) -> dict:
        """Extract relevant information from section text"""
        
        # Truncate if too long
        if len(text) > 10000:
            text = text[:5000] + "\n...[truncated]...\n" + text[-5000:]
        
        prompt = f"""Extract information relevant to answering this question.

Question: {question}

Section Text:
{text}

Return JSON:
{{
  "found_relevant": true/false,
  "excerpt": "the specific relevant text (quote exactly)",
  "summary": "what this tells us about the answer",
  "follow_refs": true/false  // should we follow cross-references?
}}"""

        return await self.llm.complete_json(prompt)
    
    async def _generate_answer(
        self,
        question: str,
        gathered_info: list,
        config: AgenticQAConfig,
    ) -> dict:
        """Generate final answer from gathered information"""
        
        if not gathered_info:
            return {
                "answer": "I could not find information to answer this question in the document.",
                "confidence": 0.0,
            }
        
        context = "Relevant information from the document:\n\n"
        for node_id, title, excerpt in gathered_info:
            context += f"From [{node_id}] {title}:\n{excerpt}\n\n"
        
        prompt = f"""Answer this question based on the document excerpts.

Question: {question}

{context}

Provide a clear, accurate answer based only on the information above.
If the information is incomplete, acknowledge what's missing.

Return JSON:
{{
  "answer": "your complete answer",
  "confidence": 0.0-1.0  // how confident based on evidence quality
}}"""

        return await self.llm.complete_json(prompt)
    
    def _get_structure_view(self, visited: list[str]) -> str:
        """Get structure view with visited markers"""
        
        def node_to_text(node: TreeNode, depth: int = 0) -> str:
            indent = "  " * depth
            visited_mark = " [visited]" if node.node_id in visited else ""
            summary = f" - {node.summary[:80]}..." if node.summary else ""
            
            lines = [f"{indent}[{node.node_id}] {node.title}{summary}{visited_mark}"]
            for child in node.children:
                lines.append(node_to_text(child, depth + 1))
            
            return "\n".join(lines)
        
        return "\n".join(node_to_text(n) for n in self.doc_index.structure)
    
    def _get_structure_summary(self) -> str:
        """Get concise structure summary"""
        
        def summarize_node(node: TreeNode, depth: int = 0) -> str:
            if depth > 2:  # Limit depth
                return ""
            indent = "  " * depth
            lines = [f"{indent}- {node.title}"]
            for child in node.children[:5]:  # Limit children shown
                child_text = summarize_node(child, depth + 1)
                if child_text:
                    lines.append(child_text)
            if len(node.children) > 5:
                lines.append(f"{indent}  ... and {len(node.children) - 5} more")
            return "\n".join(lines)
        
        return "\n".join(summarize_node(n) for n in self.doc_index.structure)
```

### 8. Provenance Extractor (`provenance.py`)

```python
"""
Provenance Extraction - exhaustive scan to find all evidence related to a topic.

Unlike Agentic QA (which stops when it has enough), Provenance Extraction
scans ALL nodes to find EVERY piece of evidence.

Use cases:
- Compliance: Find all mentions of a regulatory topic
- Due diligence: Extract all risk-related content
- Research: Gather all evidence about a specific theme
"""

from dataclasses import dataclass
from typing import Optional
import asyncio
from .models import DocumentIndex, TreeNode, NodeMatch, ProvenanceResult
from .searcher import NodeSearcher, NodeSearchConfig
from .llm_client import LLMClient, LLMConfig
from .cache import CacheManager


@dataclass
class ProvenanceConfig:
    """Configuration for provenance extraction"""
    relevance_threshold: float = 0.6
    extract_excerpts: bool = True
    max_excerpts_per_node: int = 5
    generate_summary: bool = True
    parallel_workers: int = 5


class ProvenanceExtractor:
    """
    Exhaustive provenance extraction.
    
    Scans ALL nodes in the document to find every piece of
    evidence related to a topic.
    """
    
    def __init__(
        self,
        doc_index: DocumentIndex,
        llm_config: LLMConfig = None,
        cache_manager: CacheManager = None,
    ):
        self.doc_index = doc_index
        self.llm = LLMClient(llm_config or LLMConfig())
        self.cache = cache_manager
        self.searcher = NodeSearcher(doc_index, llm_config, cache_manager)
    
    async def extract_all(
        self,
        topic: str,
        config: ProvenanceConfig = None,
    ) -> ProvenanceResult:
        """
        Extract all evidence related to a topic.
        
        Args:
            topic: The topic/theme to extract evidence for
            config: Extraction configuration
        
        Returns:
            ProvenanceResult with all matching nodes and evidence
        """
        config = config or ProvenanceConfig()
        
        # Get all nodes
        all_nodes = self.doc_index.get_all_nodes()
        total_nodes = len(all_nodes)
        
        # Score all nodes
        search_config = NodeSearchConfig(
            relevance_threshold=config.relevance_threshold,
            max_results=total_nodes,  # No limit - we want everything
            follow_cross_refs=True,
        )
        
        matches = await self.searcher.find_related_nodes(topic, search_config)
        
        # Extract specific excerpts from matching nodes
        if config.extract_excerpts:
            await self._extract_excerpts(matches, topic, config)
        
        # Generate overall summary
        summary = None
        if config.generate_summary and matches:
            summary = await self._generate_summary(topic, matches)
        
        return ProvenanceResult(
            topic=topic,
            evidence=matches,
            total_nodes_scanned=total_nodes,
            scan_coverage=1.0,  # We scan everything
            summary=summary,
        )
    
    async def extract_by_category(
        self,
        categories: dict[str, str],
        config: ProvenanceConfig = None,
    ) -> dict[str, ProvenanceResult]:
        """
        Extract evidence for multiple categories/topics.
        
        Args:
            categories: Dict of {category_name: topic_description}
            config: Extraction configuration
        
        Returns:
            Dict of {category_name: ProvenanceResult}
        """
        config = config or ProvenanceConfig()
        
        tasks = {
            name: self.extract_all(topic, config)
            for name, topic in categories.items()
        }
        
        results = await asyncio.gather(*tasks.values())
        
        return dict(zip(tasks.keys(), results))
    
    async def _extract_excerpts(
        self,
        matches: list[NodeMatch],
        topic: str,
        config: ProvenanceConfig,
    ) -> None:
        """Extract specific relevant excerpts from each match"""
        
        semaphore = asyncio.Semaphore(config.parallel_workers)
        
        async def extract_for_node(match: NodeMatch):
            async with semaphore:
                text = self.doc_index.get_node_text(match.node.node_id)
                if not text:
                    return
                
                prompt = f"""Extract the specific text excerpts from this section that are relevant to the topic.

Topic: {topic}

Section: {match.node.title}
Section Text:
{text[:8000]}

Find up to {config.max_excerpts_per_node} distinct excerpts that contain evidence related to the topic.
Quote the exact text.

Return JSON:
{{
  "excerpts": [
    "exact quote 1...",
    "exact quote 2..."
  ]
}}"""

                try:
                    result = await self.llm.complete_json(prompt)
                    match.matched_excerpts = result.get("excerpts", [])
                except Exception:
                    match.matched_excerpts = []
        
        await asyncio.gather(*[extract_for_node(m) for m in matches])
    
    async def _generate_summary(
        self,
        topic: str,
        matches: list[NodeMatch],
    ) -> str:
        """Generate summary of all evidence found"""
        
        # Build evidence summary
        evidence_text = ""
        for match in matches[:20]:  # Limit for context window
            evidence_text += f"\n[{match.node.node_id}] {match.node.title} (relevance: {match.relevance_score:.2f}):\n"
            for excerpt in match.matched_excerpts[:2]:
                evidence_text += f"  - {excerpt[:300]}...\n"
        
        prompt = f"""Summarize the evidence found in this document about the topic.

Topic: {topic}

Evidence found:
{evidence_text}

Provide a comprehensive summary of:
1. Key findings related to the topic
2. How extensively the topic is covered
3. Any notable patterns or themes

Summary:"""

        return await self.llm.complete(prompt)
```

---

## Project Structure

```
documentindex/
├── __init__.py           # Public API and exports
├── models.py             # Data models (TreeNode, DocumentIndex, etc.)
├── chunker.py            # Text chunking with semantic boundaries
├── detector.py           # Financial document type detection
├── indexer.py            # Tree structure builder
├── metadata.py           # Metadata extraction
├── cross_ref.py          # Cross-reference detection and resolution
├── llm_client.py         # Multi-provider LLM client (litellm)
├── cache.py              # Caching layer (memory, file, Redis)
├── searcher.py           # Node search (foundation for retrieval)
├── agentic_qa.py         # Agentic question answering
├── provenance.py         # Provenance extraction
└── utils.py              # Utility functions
```

---

## Dependencies

```toml
[project]
name = "documentindex"
version = "0.1.0"
description = "Lightweight hierarchical tree index for financial documents"
requires-python = ">=3.10"

dependencies = [
    "litellm>=1.40.0",     # Multi-provider LLM support (OpenAI, Anthropic, Bedrock, Azure)
    "tiktoken>=0.5.0",     # Token counting
]

[project.optional-dependencies]
cache = [
    "redis>=5.0.0",        # Redis cache backend
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
]
```

---

## Litellm Provider Support

**litellm** provides unified access to 100+ LLM providers. Key supported providers for your use case:

| Provider | Model Format | Auth Method |
|----------|--------------|-------------|
| OpenAI | `gpt-4o`, `openai/gpt-4-turbo` | `OPENAI_API_KEY` env var |
| Anthropic | `anthropic/claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` env var |
| AWS Bedrock | `bedrock/anthropic.claude-3-sonnet-20240229-v1:0` | AWS credentials (env/file/IAM) |
| Azure OpenAI | `azure/deployment-name` | `AZURE_API_KEY` + config |
| Ollama | `ollama/llama2` | Local (no auth) |

---

## Summary: How Use Cases Are Achieved

| Use Case | Component | Method |
|----------|-----------|--------|
| **1. Tree Index Generation** | `DocumentIndexer` | `indexer.index(text)` |
| **2. Node Search with Text Mapping** | `NodeSearcher` | `searcher.find_related_nodes(query)` |
| **3a. Question Answering (Agentic)** | `AgenticQA` | `qa.answer(question)` |
| **3b. Provenance Extraction** | `ProvenanceExtractor` | `extractor.extract_all(topic)` |

All components include:
- ✅ **Caching**: Via `CacheManager` (memory/file/Redis)
- ✅ **Metadata Extraction**: Via `MetadataExtractor`
- ✅ **Cross-Reference Resolution**: Via `CrossReferenceResolver`
- ✅ **Multi-provider LLM**: Via `litellm` (AWS Bedrock, Azure, OpenAI, Anthropic, etc.)
- ✅ **Streaming Support**: For large inputs/outputs with progress callbacks

---

## Streaming Support

For large financial documents, both LLM inputs and outputs can be substantial. DocumentIndex provides comprehensive streaming support to handle this efficiently.

### Design Principles

1. **Streaming Output**: LLM responses are streamed chunk-by-chunk for faster time-to-first-token
2. **Chunked Input Processing**: Large documents are processed in batches with progress tracking
3. **Progress Callbacks**: All long-running operations support progress reporting
4. **Async Generators**: Results are yielded incrementally where appropriate
5. **Memory Efficiency**: Avoid loading entire responses into memory

### Streaming Models (`streaming.py`)

```python
"""
Streaming support for DocumentIndex.

Provides:
- Streaming LLM responses
- Progress tracking for long operations
- Chunked processing for large documents
- Async generators for incremental results
"""

from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional, Any, TypeVar
from enum import Enum
import asyncio
from datetime import datetime


# ============================================================================
# Progress Tracking
# ============================================================================

class OperationType(Enum):
    """Types of long-running operations"""
    INDEXING = "indexing"
    SEARCHING = "searching"
    QA = "question_answering"
    PROVENANCE = "provenance_extraction"
    METADATA = "metadata_extraction"
    CROSS_REF = "cross_reference_resolution"


@dataclass
class ProgressUpdate:
    """Progress update for long-running operations"""
    operation: OperationType
    current_step: int
    total_steps: int
    step_name: str
    message: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    
    @property
    def progress_pct(self) -> float:
        """Progress as percentage (0-100)"""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100
    
    @property
    def is_complete(self) -> bool:
        return self.current_step >= self.total_steps


# Type for progress callback
ProgressCallback = Callable[[ProgressUpdate], None]
AsyncProgressCallback = Callable[[ProgressUpdate], asyncio.Future]


@dataclass
class StreamChunk:
    """A chunk of streamed LLM response"""
    content: str
    is_complete: bool = False
    token_count: int = 0
    accumulated_content: str = ""  # Full content so far


@dataclass
class StreamingQAResult:
    """Streaming result from agentic QA"""
    # Streamed answer chunks
    answer_stream: AsyncIterator[StreamChunk]
    
    # Available after streaming completes
    question: str = ""
    confidence: float = 0.0
    citations: list = field(default_factory=list)
    reasoning_trace: list = field(default_factory=list)
    nodes_visited: list = field(default_factory=list)


@dataclass  
class StreamingProvenanceResult:
    """Streaming result from provenance extraction"""
    topic: str
    
    # Yields NodeMatch objects as they're found
    evidence_stream: AsyncIterator['NodeMatch']
    
    # Available after streaming completes
    total_nodes_scanned: int = 0
    scan_coverage: float = 0.0
```

### Enhanced LLM Client with Streaming (`llm_client.py` additions)

```python
class LLMClient:
    """
    Enhanced LLM client with streaming support.
    
    Streaming is supported for all major providers via litellm:
    - OpenAI: Full streaming support
    - Anthropic: Full streaming support  
    - AWS Bedrock: Streaming via bedrock runtime
    - Azure OpenAI: Full streaming support
    - Ollama: Full streaming support
    """
    
    # ... (previous __init__ and complete methods) ...
    
    async def complete_stream(
        self,
        prompt: str,
        system_prompt: str = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream completion from LLM.
        
        Yields StreamChunk objects as tokens arrive.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            response_format: "text" or "json"
        
        Yields:
            StreamChunk with incremental content
        
        Example:
            async for chunk in client.complete_stream("Explain quantum computing"):
                print(chunk.content, end="", flush=True)
                if chunk.is_complete:
                    print(f"\n\nTotal tokens: {chunk.token_count}")
        """
        litellm = await self._get_litellm()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        user_content = prompt
        if response_format == "json":
            user_content += "\n\nRespond with valid JSON only."
        
        messages.append({"role": "user", "content": user_content})
        
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": True,  # Enable streaming
        }
        kwargs.update(self.config.provider_config)
        
        accumulated = ""
        token_count = 0
        
        try:
            response = await litellm.acompletion(**kwargs)
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    accumulated += content
                    token_count += 1  # Approximate
                    
                    yield StreamChunk(
                        content=content,
                        is_complete=False,
                        token_count=token_count,
                        accumulated_content=accumulated,
                    )
            
            # Final chunk
            yield StreamChunk(
                content="",
                is_complete=True,
                token_count=token_count,
                accumulated_content=accumulated,
            )
            
        except Exception as e:
            # Yield error as final chunk
            yield StreamChunk(
                content=f"\n[Error: {str(e)}]",
                is_complete=True,
                token_count=token_count,
                accumulated_content=accumulated + f"\n[Error: {str(e)}]",
            )
    
    async def complete_stream_json(
        self,
        prompt: str,
        system_prompt: str = None,
    ) -> tuple[AsyncIterator[StreamChunk], asyncio.Future[dict]]:
        """
        Stream completion and parse JSON when complete.
        
        Returns:
            Tuple of (stream iterator, future that resolves to parsed JSON)
        
        Example:
            stream, json_future = await client.complete_stream_json(prompt)
            
            async for chunk in stream:
                print(chunk.content, end="")
            
            result = await json_future
            print(f"Parsed: {result}")
        """
        accumulated = []
        json_future = asyncio.get_event_loop().create_future()
        
        async def stream_and_collect():
            async for chunk in self.complete_stream(prompt, system_prompt, "json"):
                accumulated.append(chunk.content)
                yield chunk
                
                if chunk.is_complete:
                    try:
                        full_text = "".join(accumulated)
                        parsed = self._extract_json(full_text)
                        json_future.set_result(parsed)
                    except Exception as e:
                        json_future.set_exception(e)
        
        return stream_and_collect(), json_future


    # -------------------------------------------------------------------------
    # Chunked Input Processing (for large prompts)
    # -------------------------------------------------------------------------
    
    async def complete_chunked(
        self,
        prompt_parts: list[str],
        system_prompt: str = None,
        combine_strategy: Literal["concat", "summarize", "last"] = "concat",
        progress_callback: ProgressCallback = None,
    ) -> str:
        """
        Process large input by chunking into multiple LLM calls.
        
        Useful when input exceeds context window.
        
        Args:
            prompt_parts: List of prompt parts to process
            system_prompt: System prompt for all calls
            combine_strategy: How to combine results
                - "concat": Concatenate all results
                - "summarize": Summarize combined results
                - "last": Use only the last result (for iterative refinement)
            progress_callback: Called after each chunk
        
        Returns:
            Combined result string
        """
        results = []
        total = len(prompt_parts)
        
        for i, part in enumerate(prompt_parts):
            result = await self.complete(part, system_prompt)
            results.append(result)
            
            if progress_callback:
                progress_callback(ProgressUpdate(
                    operation=OperationType.INDEXING,
                    current_step=i + 1,
                    total_steps=total,
                    step_name=f"Processing chunk {i + 1}/{total}",
                    message=f"Processed {len(part)} characters",
                ))
        
        if combine_strategy == "concat":
            return "\n\n".join(results)
        elif combine_strategy == "last":
            return results[-1] if results else ""
        elif combine_strategy == "summarize":
            combined = "\n\n".join(results)
            summary_prompt = f"Summarize and consolidate these results:\n\n{combined}"
            return await self.complete(summary_prompt, system_prompt)
        
        return results[-1] if results else ""
```

### Streaming Document Indexer (`indexer.py` additions)

```python
class DocumentIndexer:
    """
    Document indexer with streaming and progress support.
    """
    
    async def index_with_progress(
        self,
        text: str,
        doc_name: str = "document",
        doc_type: DocumentType = None,
        progress_callback: ProgressCallback = None,
    ) -> DocumentIndex:
        """
        Build index with progress reporting.
        
        Progress steps:
        1. Chunking text
        2. Detecting document type
        3. Building structure (may have sub-steps for large docs)
        4. Generating summaries
        5. Extracting metadata
        6. Resolving cross-references
        7. Finalizing index
        
        Example:
            def on_progress(update: ProgressUpdate):
                print(f"[{update.progress_pct:.1f}%] {update.step_name}: {update.message}")
            
            index = await indexer.index_with_progress(text, progress_callback=on_progress)
        """
        total_steps = 7
        current_step = 0
        
        def report(step_name: str, message: str = ""):
            nonlocal current_step
            current_step += 1
            if progress_callback:
                progress_callback(ProgressUpdate(
                    operation=OperationType.INDEXING,
                    current_step=current_step,
                    total_steps=total_steps,
                    step_name=step_name,
                    message=message,
                ))
        
        # Step 1: Chunk text
        report("Chunking", f"Processing {len(text)} characters")
        chunks = self.chunker.chunk(text)
        
        # Step 2: Detect type
        report("Type Detection", f"Analyzing document structure")
        if doc_type is None:
            doc_type = FinancialDocDetector.detect(text, doc_name)
        
        # Step 3: Build structure
        report("Structure Building", f"Analyzing {len(chunks)} chunks")
        structure = await self._build_structure_streaming(chunks, doc_type, progress_callback)
        
        # Step 4: Generate summaries
        if self.config.generate_summaries:
            report("Summary Generation", f"Summarizing {len(structure)} sections")
            await self._generate_summaries_streaming(structure, chunks, progress_callback)
        else:
            report("Summary Generation", "Skipped")
        
        # Step 5: Extract metadata
        report("Metadata Extraction", "Extracting document metadata")
        metadata = await self.metadata_extractor.extract(text, doc_type, doc_name)
        
        # Step 6: Resolve cross-references
        report("Cross-Reference Resolution", "Finding and resolving references")
        # ... cross-ref resolution ...
        
        # Step 7: Finalize
        report("Finalizing", "Building final index")
        # ... create DocumentIndex ...
        
        return doc_index
    
    async def index_stream(
        self,
        text: str,
        doc_name: str = "document",
        doc_type: DocumentType = None,
    ) -> AsyncIterator[ProgressUpdate]:
        """
        Build index as an async generator yielding progress updates.
        
        The final yield contains the completed DocumentIndex in its message field.
        
        Example:
            async for update in indexer.index_stream(text):
                print(f"{update.progress_pct:.1f}%: {update.step_name}")
                if update.is_complete:
                    doc_index = update.message  # Final result
        """
        result_holder = {}
        
        def capture_progress(update: ProgressUpdate):
            # Just for tracking, actual yield happens below
            pass
        
        # Create task for indexing
        async def do_index():
            result_holder['index'] = await self.index_with_progress(
                text, doc_name, doc_type,
                progress_callback=capture_progress,
            )
        
        # Run indexing and yield progress
        total_steps = 7
        for step in range(1, total_steps + 1):
            yield ProgressUpdate(
                operation=OperationType.INDEXING,
                current_step=step,
                total_steps=total_steps,
                step_name=f"Step {step}/{total_steps}",
            )
            await asyncio.sleep(0)  # Yield control
        
        await do_index()
        
        # Final yield with result
        yield ProgressUpdate(
            operation=OperationType.INDEXING,
            current_step=total_steps,
            total_steps=total_steps,
            step_name="Complete",
            message=result_holder.get('index'),
        )
```

### Streaming Agentic QA (`agentic_qa.py` additions)

```python
class AgenticQA:
    """
    Agentic QA with streaming support.
    """
    
    async def answer_stream(
        self,
        question: str,
        config: AgenticQAConfig = None,
    ) -> StreamingQAResult:
        """
        Answer question with streaming response.
        
        Returns immediately with a StreamingQAResult containing:
        - answer_stream: Async iterator of answer chunks
        - Other fields populated after streaming completes
        
        Example:
            result = await qa.answer_stream("What was the revenue?")
            
            print("Answer: ", end="")
            async for chunk in result.answer_stream:
                print(chunk.content, end="", flush=True)
                if chunk.is_complete:
                    print()  # Newline
            
            # Now access full result
            print(f"Confidence: {result.confidence}")
            print(f"Citations: {result.citations}")
        """
        config = config or AgenticQAConfig()
        
        # Collect information (non-streaming part)
        reasoning_trace = []
        visited_nodes = []
        gathered_info = []
        
        # ... (same reasoning loop as before) ...
        
        # Create streaming answer generator
        async def generate_answer_stream() -> AsyncIterator[StreamChunk]:
            context = self._build_context(gathered_info)
            
            prompt = f"""Answer this question based on the document excerpts.

Question: {question}

{context}

Provide a clear, accurate answer:"""
            
            async for chunk in self.llm.complete_stream(prompt):
                yield chunk
        
        return StreamingQAResult(
            question=question,
            answer_stream=generate_answer_stream(),
            reasoning_trace=reasoning_trace,
            nodes_visited=visited_nodes,
            citations=[],  # Populated after stream completes
        )
    
    async def answer_with_progress(
        self,
        question: str,
        config: AgenticQAConfig = None,
        progress_callback: ProgressCallback = None,
    ) -> QAResult:
        """
        Answer question with progress reporting.
        
        Progress includes:
        - Planning phase
        - Each retrieval iteration
        - Answer generation
        
        Example:
            def on_progress(update):
                print(f"[{update.step_name}] {update.message}")
            
            result = await qa.answer_with_progress(
                "What are the risk factors?",
                progress_callback=on_progress
            )
        """
        config = config or AgenticQAConfig()
        
        def report(step: int, total: int, name: str, msg: str = ""):
            if progress_callback:
                progress_callback(ProgressUpdate(
                    operation=OperationType.QA,
                    current_step=step,
                    total_steps=total,
                    step_name=name,
                    message=msg,
                ))
        
        # Estimate total steps
        total_steps = config.max_iterations + 2  # +2 for plan and answer
        
        report(1, total_steps, "Planning", "Analyzing question and document structure")
        plan = await self._plan_search(question)
        
        # ... iteration loop with progress reporting ...
        
        report(total_steps, total_steps, "Generating Answer", "Synthesizing final answer")
        # ... generate answer ...
        
        return result
```

### Streaming Provenance Extractor (`provenance.py` additions)

```python
class ProvenanceExtractor:
    """
    Provenance extractor with streaming support.
    """
    
    async def extract_stream(
        self,
        topic: str,
        config: ProvenanceConfig = None,
    ) -> StreamingProvenanceResult:
        """
        Extract provenance with streaming results.
        
        Yields NodeMatch objects as they are found, enabling
        early access to results while scanning continues.
        
        Example:
            result = await extractor.extract_stream("climate risks")
            
            found_count = 0
            async for match in result.evidence_stream:
                found_count += 1
                print(f"Found [{match.node.node_id}] {match.node.title}")
                print(f"  Relevance: {match.relevance_score:.2f}")
            
            print(f"\nTotal found: {found_count}")
            print(f"Nodes scanned: {result.total_nodes_scanned}")
        """
        config = config or ProvenanceConfig()
        
        all_nodes = self.doc_index.get_all_nodes()
        total_nodes = len(all_nodes)
        
        # Track state
        state = {
            "scanned": 0,
            "found": 0,
        }
        
        async def evidence_generator() -> AsyncIterator[NodeMatch]:
            # Process in batches for efficiency
            batch_size = config.parallel_workers
            
            for i in range(0, len(all_nodes), batch_size):
                batch = all_nodes[i:i + batch_size]
                
                # Score batch
                matches = await self._score_batch(batch, topic)
                
                # Yield relevant matches immediately
                for match in matches:
                    state["scanned"] += 1
                    
                    if match.relevance_score >= config.relevance_threshold:
                        state["found"] += 1
                        
                        # Extract excerpts if configured
                        if config.extract_excerpts:
                            await self._extract_excerpts_for_node(match, topic, config)
                        
                        yield match
        
        return StreamingProvenanceResult(
            topic=topic,
            evidence_stream=evidence_generator(),
            total_nodes_scanned=total_nodes,
        )
    
    async def extract_with_progress(
        self,
        topic: str,
        config: ProvenanceConfig = None,
        progress_callback: ProgressCallback = None,
    ) -> ProvenanceResult:
        """
        Extract provenance with progress reporting.
        
        Example:
            def on_progress(update):
                print(f"Scanned {update.current_step}/{update.total_steps} nodes")
            
            result = await extractor.extract_with_progress(
                "regulatory compliance",
                progress_callback=on_progress
            )
        """
        config = config or ProvenanceConfig()
        
        all_nodes = self.doc_index.get_all_nodes()
        total_nodes = len(all_nodes)
        evidence = []
        
        batch_size = config.parallel_workers
        
        for i in range(0, len(all_nodes), batch_size):
            batch = all_nodes[i:i + batch_size]
            matches = await self._score_batch(batch, topic)
            
            for match in matches:
                if match.relevance_score >= config.relevance_threshold:
                    evidence.append(match)
            
            if progress_callback:
                progress_callback(ProgressUpdate(
                    operation=OperationType.PROVENANCE,
                    current_step=min(i + batch_size, total_nodes),
                    total_steps=total_nodes,
                    step_name="Scanning nodes",
                    message=f"Found {len(evidence)} relevant nodes so far",
                ))
        
        return ProvenanceResult(
            topic=topic,
            evidence=evidence,
            total_nodes_scanned=total_nodes,
            scan_coverage=1.0,
        )
```

### Usage Examples with Streaming

```python
import asyncio
from documentindex import (
    DocumentIndexer, AgenticQA, ProvenanceExtractor,
    LLMConfig, ProgressUpdate
)

# ============================================================================
# Example 1: Index with Progress Bar
# ============================================================================

async def index_with_progress_bar(text: str):
    """Show progress bar during indexing"""
    
    def show_progress(update: ProgressUpdate):
        bar_width = 40
        filled = int(bar_width * update.progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        print(f"\r[{bar}] {update.progress_pct:.1f}% - {update.step_name}", end="")
        if update.is_complete:
            print()  # Newline at end
    
    indexer = DocumentIndexer()
    doc_index = await indexer.index_with_progress(
        text,
        doc_name="SEC_10K",
        progress_callback=show_progress,
    )
    return doc_index


# ============================================================================
# Example 2: Streaming QA Response
# ============================================================================

async def streaming_qa(doc_index, question: str):
    """Stream answer as it's generated"""
    
    qa = AgenticQA(doc_index)
    result = await qa.answer_stream(question)
    
    print(f"Question: {question}")
    print("Answer: ", end="")
    
    full_answer = ""
    async for chunk in result.answer_stream:
        print(chunk.content, end="", flush=True)
        full_answer = chunk.accumulated_content
        
        if chunk.is_complete:
            print("\n")  # Newlines at end
    
    print(f"Confidence: {result.confidence}")
    print(f"Nodes visited: {len(result.nodes_visited)}")
    
    return full_answer


# ============================================================================
# Example 3: Real-time Provenance Discovery
# ============================================================================

async def realtime_provenance(doc_index, topic: str):
    """Show evidence as it's found"""
    
    extractor = ProvenanceExtractor(doc_index)
    result = await extractor.extract_stream(topic)
    
    print(f"Searching for: {topic}")
    print("-" * 50)
    
    count = 0
    async for match in result.evidence_stream:
        count += 1
        print(f"\n[{count}] Found: {match.node.title}")
        print(f"    Relevance: {match.relevance_score:.2f}")
        print(f"    Reason: {match.match_reason}")
        
        for excerpt in match.matched_excerpts[:2]:
            print(f"    > {excerpt[:100]}...")
    
    print(f"\n{'=' * 50}")
    print(f"Total matches: {count} / {result.total_nodes_scanned} nodes scanned")


# ============================================================================
# Example 4: Progress Callback with Logging
# ============================================================================

import logging

async def index_with_logging(text: str):
    """Log progress to standard logger"""
    
    logger = logging.getLogger("documentindex")
    
    def log_progress(update: ProgressUpdate):
        logger.info(
            f"[{update.operation.value}] "
            f"{update.progress_pct:.1f}% - "
            f"{update.step_name}: {update.message}"
        )
    
    indexer = DocumentIndexer()
    return await indexer.index_with_progress(
        text,
        progress_callback=log_progress,
    )


# ============================================================================
# Example 5: Combine Streaming with Caching
# ============================================================================

async def cached_streaming_qa(doc_index, question: str, cache):
    """
    Check cache first, stream if not cached.
    Cache the result after streaming completes.
    """
    
    # Check cache
    cached_result = await cache.get_search_result(doc_index.doc_id, question)
    if cached_result:
        print("[From cache]")
        print(f"Answer: {cached_result.answer}")
        return cached_result
    
    # Stream new answer
    qa = AgenticQA(doc_index)
    result = await qa.answer_stream(question)
    
    print("[Generating]")
    print("Answer: ", end="")
    
    async for chunk in result.answer_stream:
        print(chunk.content, end="", flush=True)
    print()
    
    # Cache the completed result
    await cache.set_search_result(doc_index.doc_id, question, result)
    
    return result
```

### Project Structure (Updated)

```
documentindex/
├── __init__.py           # Public API and exports
├── models.py             # Data models (TreeNode, DocumentIndex, etc.)
├── streaming.py          # NEW: Streaming models and utilities
├── chunker.py            # Text chunking with semantic boundaries
├── detector.py           # Financial document type detection
├── indexer.py            # Tree structure builder (with streaming)
├── metadata.py           # Metadata extraction
├── cross_ref.py          # Cross-reference detection and resolution
├── llm_client.py         # Multi-provider LLM client (with streaming)
├── cache.py              # Caching layer (memory, file, Redis)
├── searcher.py           # Node search (foundation for retrieval)
├── agentic_qa.py         # Agentic question answering (with streaming)
├── provenance.py         # Provenance extraction (with streaming)
└── utils.py              # Utility functions
```

### Dependencies (Updated)

```toml
[project]
name = "documentindex"
version = "0.1.0"
description = "Lightweight hierarchical tree index for financial documents"
requires-python = ">=3.10"

dependencies = [
    "litellm>=1.40.0",     # Multi-provider LLM support with streaming
    "tiktoken>=0.5.0",     # Token counting
    "aiofiles>=23.0.0",    # Async file operations (for streaming to disk)
]

[project.optional-dependencies]
cache = [
    "redis>=5.0.0",        # Redis cache backend
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
]
```

### Summary: Streaming Capabilities

| Component | Streaming Method | Use Case |
|-----------|------------------|----------|
| `LLMClient` | `complete_stream()` | Stream LLM responses token-by-token |
| `LLMClient` | `complete_chunked()` | Process large inputs in batches |
| `DocumentIndexer` | `index_with_progress()` | Progress callbacks during indexing |
| `DocumentIndexer` | `index_stream()` | Async generator with progress yields |
| `AgenticQA` | `answer_stream()` | Stream answer as it's generated |
| `AgenticQA` | `answer_with_progress()` | Progress callbacks during reasoning |
| `ProvenanceExtractor` | `extract_stream()` | Yield evidence as found |
| `ProvenanceExtractor` | `extract_with_progress()` | Progress callbacks during scan |

All streaming methods support:
- ✅ Early termination (consumer can stop early)
- ✅ Progress tracking via callbacks
- ✅ Memory efficiency (no full response buffering)
- ✅ Integration with caching (cache completed results)
- ✅ Error handling (errors surfaced through stream)
