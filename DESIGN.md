# DocumentIndex: Lightweight Hierarchical Tree Index for Financial Documents

## Executive Summary

**DocumentIndex** is a lightweight, text-focused alternative to PageIndex designed specifically for building hierarchical tree indexes for long financial documents. Unlike PageIndex which handles PDF parsing and vision-based approaches, DocumentIndex operates on pre-extracted text, making it simpler, faster, and more suitable for financial document workflows where text is already available (e.g., from edgartools for SEC filings).

## Design Goals

### Primary Goals
1. **Lightweight**: Minimal dependencies, no PDF/vision processing
2. **Text-First**: Accepts plain text input, works with any document conversion pipeline
3. **Financial Document Optimized**: Heuristics tuned for SEC filings, earnings calls, research reports
4. **Reasoning-Based Retrieval**: Hierarchical tree structure for LLM navigation (not vector search)
5. **Async-Native**: Built for concurrent processing of large document sets
6. **LLM-Agnostic**: Support multiple LLM providers (OpenAI, Anthropic, local models via litellm)

### Non-Goals (Simplifications from PageIndex)
- No PDF parsing (input is text)
- No vision/OCR support
- No image extraction
- No complex TOC page detection (works on structured text)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DocumentIndex                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Chunker    │───▶│  Indexer     │───▶│   TreeBuilder        │  │
│  │  (text→pages)│    │ (structure)  │    │ (hierarchical index) │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│         │                   │                       │               │
│         ▼                   ▼                       ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ FinancialDoc │    │  LLMClient   │    │    DocumentTree      │  │
│  │   Detector   │    │ (multi-prov) │    │     (output)         │  │
│  └──────────────┘    └──────────────┘    └──────────────────────┘  │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│                        Retriever Module                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  TreeSearcher: Reasoning-based navigation over tree index    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Document Model (`models.py`)

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class DocumentType(Enum):
    """Financial document types with specific parsing heuristics"""
    SEC_10K = "10-K"
    SEC_10Q = "10-Q"
    SEC_8K = "8-K"
    SEC_DEF14A = "DEF 14A"
    SEC_S1 = "S-1"
    EARNINGS_CALL = "earnings_call"
    RESEARCH_REPORT = "research_report"
    FINANCIAL_NEWS = "financial_news"
    GENERIC = "generic"

@dataclass
class TreeNode:
    """A node in the hierarchical document tree"""
    node_id: str
    title: str
    start_index: int  # Start position in chunks
    end_index: int    # End position in chunks
    level: int = 0    # Hierarchy level (0 = root level)
    summary: Optional[str] = None
    children: list['TreeNode'] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        result = {
            "node_id": self.node_id,
            "title": self.title,
            "start_index": self.start_index,
            "end_index": self.end_index,
        }
        if self.summary:
            result["summary"] = self.summary
        if self.children:
            result["nodes"] = [child.to_dict() for child in self.children]
        if self.metadata:
            result["metadata"] = self.metadata
        return result

@dataclass
class DocumentIndex:
    """Complete document index with tree structure"""
    doc_name: str
    doc_type: DocumentType
    description: Optional[str] = None
    total_chunks: int = 0
    structure: list[TreeNode] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)  # Original text chunks
    
    def to_dict(self) -> dict:
        return {
            "doc_name": self.doc_name,
            "doc_type": self.doc_type.value,
            "doc_description": self.description,
            "total_chunks": self.total_chunks,
            "structure": [node.to_dict() for node in self.structure]
        }
    
    def get_chunk_text(self, start: int, end: int) -> str:
        """Get text content for a range of chunks"""
        return "\n".join(self.chunks[start:end])
    
    def get_node_text(self, node_id: str) -> Optional[str]:
        """Get text content for a specific node"""
        node = self._find_node(node_id, self.structure)
        if node:
            return self.get_chunk_text(node.start_index, node.end_index)
        return None
```

### 2. Text Chunker (`chunker.py`)

The chunker converts long text into manageable chunks while preserving semantic boundaries.

```python
from dataclasses import dataclass
from typing import Iterator
import re

@dataclass
class ChunkerConfig:
    """Configuration for text chunking"""
    max_chunk_tokens: int = 2000      # Max tokens per chunk
    overlap_tokens: int = 100          # Overlap between chunks
    respect_paragraphs: bool = True    # Try to break on paragraph boundaries
    respect_sections: bool = True      # Try to break on section boundaries

class TextChunker:
    """Chunks text while preserving semantic boundaries"""
    
    # Financial document section patterns
    SECTION_PATTERNS = [
        r'^(?:PART|Part)\s+[IVX]+',           # PART I, PART II, etc.
        r'^(?:ITEM|Item)\s+\d+',              # ITEM 1, ITEM 1A, etc.
        r'^(?:SECTION|Section)\s+\d+',        # SECTION 1, SECTION 2
        r'^\d+\.\s+[A-Z][A-Za-z\s]+$',        # 1. Introduction
        r'^[A-Z][A-Z\s]+:?\s*$',              # ALL CAPS HEADERS
        r'^(?:Note|NOTE)\s+\d+',              # Note 1, NOTE 2
        r'^(?:Exhibit|EXHIBIT)\s+\d+',        # Exhibit 10.1
    ]
    
    def __init__(self, config: ChunkerConfig = None):
        self.config = config or ChunkerConfig()
        self._section_regex = re.compile(
            '|'.join(f'({p})' for p in self.SECTION_PATTERNS),
            re.MULTILINE
        )
    
    def chunk(self, text: str) -> list[str]:
        """Split text into chunks respecting semantic boundaries"""
        # First, split into paragraphs
        paragraphs = self._split_paragraphs(text)
        
        # Then, group paragraphs into chunks
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            
            # Check if this paragraph starts a new section
            is_section_start = self._is_section_header(para)
            
            # Decide whether to start a new chunk
            should_break = (
                (current_tokens + para_tokens > self.config.max_chunk_tokens) or
                (is_section_start and self.config.respect_sections and current_chunk)
            )
            
            if should_break and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                # Add overlap from previous chunk
                overlap_paras = self._get_overlap_paragraphs(current_chunk)
                current_chunk = overlap_paras
                current_tokens = sum(self._estimate_tokens(p) for p in overlap_paras)
            
            current_chunk.append(para)
            current_tokens += para_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks
    
    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs"""
        # Split on double newlines or more
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: ~4 chars per token)"""
        return len(text) // 4
    
    def _is_section_header(self, text: str) -> bool:
        """Check if text looks like a section header"""
        first_line = text.split('\n')[0].strip()
        return bool(self._section_regex.match(first_line))
    
    def _get_overlap_paragraphs(self, paragraphs: list[str]) -> list[str]:
        """Get paragraphs for overlap from the end of chunk"""
        if not self.config.overlap_tokens:
            return []
        
        overlap = []
        tokens = 0
        for para in reversed(paragraphs):
            para_tokens = self._estimate_tokens(para)
            if tokens + para_tokens > self.config.overlap_tokens:
                break
            overlap.insert(0, para)
            tokens += para_tokens
        return overlap
```

### 3. Financial Document Detector (`detector.py`)

Automatically detects document type for optimized parsing.

```python
import re
from .models import DocumentType

class FinancialDocDetector:
    """Detects the type of financial document from text content"""
    
    SEC_FILING_PATTERNS = {
        DocumentType.SEC_10K: [
            r'FORM\s+10-K',
            r'ANNUAL\s+REPORT\s+PURSUANT\s+TO\s+SECTION\s+13',
            r'For the fiscal year ended',
        ],
        DocumentType.SEC_10Q: [
            r'FORM\s+10-Q',
            r'QUARTERLY\s+REPORT\s+PURSUANT\s+TO\s+SECTION\s+13',
            r'For the quarterly period ended',
        ],
        DocumentType.SEC_8K: [
            r'FORM\s+8-K',
            r'CURRENT\s+REPORT\s+PURSUANT\s+TO\s+SECTION\s+13',
        ],
        DocumentType.SEC_DEF14A: [
            r'DEF\s*14A',
            r'PROXY\s+STATEMENT',
            r'SCHEDULE\s+14A',
        ],
        DocumentType.SEC_S1: [
            r'FORM\s+S-1',
            r'REGISTRATION\s+STATEMENT',
        ],
    }
    
    EARNINGS_PATTERNS = [
        r'(?:earnings|quarterly)\s+(?:call|conference)',
        r'Q[1-4]\s+\d{4}\s+(?:earnings|results)',
        r'operator|participants|presentation',
    ]
    
    RESEARCH_PATTERNS = [
        r'(?:buy|sell|hold|overweight|underweight)\s+rating',
        r'price\s+target',
        r'(?:analyst|equity\s+research)',
    ]
    
    @classmethod
    def detect(cls, text: str, filename: str = None) -> DocumentType:
        """Detect document type from text content and optional filename"""
        # Check first ~5000 chars for efficiency
        sample = text[:5000].upper()
        
        # Check SEC filing patterns
        for doc_type, patterns in cls.SEC_FILING_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, sample, re.IGNORECASE):
                    return doc_type
        
        # Check earnings call patterns
        earnings_matches = sum(
            1 for p in cls.EARNINGS_PATTERNS 
            if re.search(p, sample, re.IGNORECASE)
        )
        if earnings_matches >= 2:
            return DocumentType.EARNINGS_CALL
        
        # Check research report patterns
        research_matches = sum(
            1 for p in cls.RESEARCH_PATTERNS 
            if re.search(p, sample, re.IGNORECASE)
        )
        if research_matches >= 2:
            return DocumentType.RESEARCH_REPORT
        
        # Try to infer from filename
        if filename:
            filename_lower = filename.lower()
            if '10-k' in filename_lower or '10k' in filename_lower:
                return DocumentType.SEC_10K
            if '10-q' in filename_lower or '10q' in filename_lower:
                return DocumentType.SEC_10Q
            if '8-k' in filename_lower or '8k' in filename_lower:
                return DocumentType.SEC_8K
        
        return DocumentType.GENERIC
```

### 4. LLM Client (`llm_client.py`)

Multi-provider LLM client using litellm for flexibility.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import asyncio
import json
import re

@dataclass
class LLMConfig:
    """LLM configuration"""
    model: str = "gpt-4o"
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay: float = 1.0

class LLMClient:
    """Async LLM client supporting multiple providers via litellm"""
    
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig()
        self._litellm = None
    
    async def _get_litellm(self):
        """Lazy import litellm"""
        if self._litellm is None:
            import litellm
            self._litellm = litellm
        return self._litellm
    
    async def complete(self, prompt: str, system_prompt: str = None) -> str:
        """Get completion from LLM"""
        litellm = await self._get_litellm()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(self.config.max_retries):
            try:
                response = await litellm.acompletion(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
    
    async def complete_json(self, prompt: str, system_prompt: str = None) -> dict:
        """Get JSON completion from LLM"""
        response = await self.complete(prompt, system_prompt)
        return self._extract_json(response)
    
    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract JSON from LLM response"""
        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            text = json_match.group(1)
        
        # Clean up common issues
        text = text.strip()
        text = re.sub(r',\s*}', '}', text)  # Remove trailing commas
        text = re.sub(r',\s*]', ']', text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object or array
            for start, end in [('{', '}'), ('[', ']')]:
                start_idx = text.find(start)
                end_idx = text.rfind(end)
                if start_idx != -1 and end_idx != -1:
                    try:
                        return json.loads(text[start_idx:end_idx + 1])
                    except:
                        continue
            return {}
```

### 5. Tree Indexer (`indexer.py`)

The core component that builds hierarchical tree structure from document text.

```python
from dataclasses import dataclass
from typing import Optional
import asyncio
from .models import TreeNode, DocumentIndex, DocumentType
from .chunker import TextChunker, ChunkerConfig
from .detector import FinancialDocDetector
from .llm_client import LLMClient, LLMConfig

@dataclass
class IndexerConfig:
    """Configuration for document indexer"""
    llm_config: LLMConfig = None
    chunker_config: ChunkerConfig = None
    max_nodes_per_level: int = 20
    generate_summaries: bool = True
    generate_description: bool = True
    max_concurrent_requests: int = 5

class DocumentIndexer:
    """Builds hierarchical tree index from document text"""
    
    def __init__(self, config: IndexerConfig = None):
        self.config = config or IndexerConfig()
        self.llm = LLMClient(self.config.llm_config or LLMConfig())
        self.chunker = TextChunker(self.config.chunker_config or ChunkerConfig())
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
    
    async def index(
        self,
        text: str,
        doc_name: str = "document",
        doc_type: DocumentType = None,
    ) -> DocumentIndex:
        """Build hierarchical index for document text"""
        
        # Detect document type if not provided
        if doc_type is None:
            doc_type = FinancialDocDetector.detect(text, doc_name)
        
        # Chunk the text
        chunks = self.chunker.chunk(text)
        
        # Build initial structure
        structure = await self._build_structure(chunks, doc_type)
        
        # Generate summaries if configured
        if self.config.generate_summaries:
            await self._generate_summaries(structure, chunks)
        
        # Create document index
        doc_index = DocumentIndex(
            doc_name=doc_name,
            doc_type=doc_type,
            total_chunks=len(chunks),
            structure=structure,
            chunks=chunks,
        )
        
        # Generate description if configured
        if self.config.generate_description:
            doc_index.description = await self._generate_description(structure)
        
        return doc_index
    
    async def _build_structure(
        self,
        chunks: list[str],
        doc_type: DocumentType,
    ) -> list[TreeNode]:
        """Build tree structure from chunks using LLM"""
        
        # Prepare chunked text with position markers
        marked_text = self._prepare_marked_text(chunks)
        
        # Get structure from LLM
        prompt = self._get_structure_prompt(marked_text, doc_type)
        response = await self.llm.complete_json(prompt)
        
        # Parse response into tree nodes
        if isinstance(response, list):
            structure = response
        else:
            structure = response.get('structure', response.get('sections', []))
        
        # Convert to TreeNode objects
        nodes = self._parse_structure(structure)
        
        # Validate and fix indices
        nodes = self._validate_indices(nodes, len(chunks))
        
        # Recursively process large nodes
        nodes = await self._process_large_nodes(nodes, chunks, doc_type)
        
        return nodes
    
    def _prepare_marked_text(self, chunks: list[str], max_chars: int = 100000) -> str:
        """Prepare text with chunk position markers"""
        parts = []
        total_chars = 0
        
        for i, chunk in enumerate(chunks):
            marker = f"<chunk_{i}>"
            # Truncate chunk if needed for context window
            truncated = chunk[:2000] if len(chunk) > 2000 else chunk
            part = f"{marker}\n{truncated}\n{marker}\n"
            
            if total_chars + len(part) > max_chars:
                # Add summary for remaining
                parts.append(f"\n[... {len(chunks) - i} more chunks ...]\n")
                break
            
            parts.append(part)
            total_chars += len(part)
        
        return "\n".join(parts)
    
    def _get_structure_prompt(self, text: str, doc_type: DocumentType) -> str:
        """Generate prompt for structure extraction"""
        
        doc_hints = {
            DocumentType.SEC_10K: "Look for: PART I/II/III/IV, Items 1-15, Risk Factors, MD&A, Financial Statements, Notes",
            DocumentType.SEC_10Q: "Look for: PART I/II, Items 1-4, Financial Statements, Notes, MD&A",
            DocumentType.SEC_8K: "Look for: Item numbers (1.01-9.01), Exhibits, Signatures",
            DocumentType.EARNINGS_CALL: "Look for: Operator intro, Prepared Remarks, Q&A Session, speaker sections",
            DocumentType.RESEARCH_REPORT: "Look for: Executive Summary, Investment Thesis, Valuation, Risks, Financial Analysis",
        }.get(doc_type, "Extract the main sections and subsections")
        
        return f'''You are an expert at extracting hierarchical document structure.
Analyze this {doc_type.value} document and extract its hierarchical structure.

{doc_hints}

Document text (with chunk markers):
{text}

Return a JSON array of sections. Each section should have:
- "structure": hierarchical index like "1", "1.1", "1.2", "2", etc.
- "title": the section title (use original title from document)
- "chunk_index": the <chunk_N> marker where this section starts

Example format:
[
  {{"structure": "1", "title": "PART I", "chunk_index": 0}},
  {{"structure": "1.1", "title": "Item 1. Business", "chunk_index": 1}},
  {{"structure": "1.2", "title": "Item 1A. Risk Factors", "chunk_index": 5}},
  ...
]

Return only the JSON array. Extract all major sections and subsections.'''
    
    def _parse_structure(self, structure: list, parent_level: int = 0) -> list[TreeNode]:
        """Convert structure list to TreeNode objects"""
        nodes = []
        node_counter = 0
        
        for item in structure:
            if not isinstance(item, dict):
                continue
            
            # Get chunk index
            chunk_idx = item.get('chunk_index', 0)
            if isinstance(chunk_idx, str):
                # Extract number from marker like "<chunk_5>"
                import re
                match = re.search(r'\d+', chunk_idx)
                chunk_idx = int(match.group()) if match else 0
            
            # Determine level from structure string
            struct = item.get('structure', '')
            level = struct.count('.') if struct else parent_level
            
            node = TreeNode(
                node_id=str(node_counter).zfill(4),
                title=item.get('title', 'Untitled'),
                start_index=chunk_idx,
                end_index=chunk_idx + 1,  # Will be updated later
                level=level,
            )
            
            # Handle nested nodes
            if 'nodes' in item or 'subsections' in item:
                children = item.get('nodes') or item.get('subsections', [])
                node.children = self._parse_structure(children, level + 1)
            
            nodes.append(node)
            node_counter += 1
        
        return nodes
    
    def _validate_indices(self, nodes: list[TreeNode], total_chunks: int) -> list[TreeNode]:
        """Validate and fix chunk indices in tree structure"""
        
        def fix_indices(node_list: list[TreeNode], max_end: int):
            for i, node in enumerate(node_list):
                # Ensure start_index is valid
                node.start_index = max(0, min(node.start_index, total_chunks - 1))
                
                # Calculate end_index based on next sibling or children
                if i + 1 < len(node_list):
                    node.end_index = node_list[i + 1].start_index
                else:
                    node.end_index = max_end
                
                # Ensure end > start
                if node.end_index <= node.start_index:
                    node.end_index = min(node.start_index + 1, total_chunks)
                
                # Recursively fix children
                if node.children:
                    fix_indices(node.children, node.end_index)
        
        fix_indices(nodes, total_chunks)
        return nodes
    
    async def _process_large_nodes(
        self,
        nodes: list[TreeNode],
        chunks: list[str],
        doc_type: DocumentType,
        max_chunks_per_node: int = 10,
    ) -> list[TreeNode]:
        """Recursively process large nodes to extract sub-structure"""
        
        async def process_node(node: TreeNode) -> TreeNode:
            chunk_count = node.end_index - node.start_index
            
            # If node is large and has no children, try to extract sub-structure
            if chunk_count > max_chunks_per_node and not node.children:
                node_chunks = chunks[node.start_index:node.end_index]
                sub_structure = await self._build_structure(node_chunks, doc_type)
                
                if sub_structure:
                    # Adjust indices relative to parent
                    for sub_node in sub_structure:
                        sub_node.start_index += node.start_index
                        sub_node.end_index += node.start_index
                    node.children = sub_structure
            
            # Recursively process children
            if node.children:
                tasks = [process_node(child) for child in node.children]
                node.children = await asyncio.gather(*tasks)
            
            return node
        
        tasks = [process_node(node) for node in nodes]
        return await asyncio.gather(*tasks)
    
    async def _generate_summaries(
        self,
        nodes: list[TreeNode],
        chunks: list[str],
    ) -> None:
        """Generate summaries for all nodes"""
        
        async def summarize_node(node: TreeNode):
            async with self._semaphore:
                text = "\n".join(chunks[node.start_index:node.end_index])
                # Truncate if too long
                if len(text) > 10000:
                    text = text[:5000] + "\n...\n" + text[-5000:]
                
                prompt = f'''Summarize this section of a financial document in 2-3 sentences.
Focus on the key information and purpose of this section.

Section Title: {node.title}
Section Content:
{text}

Provide a concise summary:'''
                
                node.summary = await self.llm.complete(prompt)
            
            # Summarize children
            if node.children:
                tasks = [summarize_node(child) for child in node.children]
                await asyncio.gather(*tasks)
        
        tasks = [summarize_node(node) for node in nodes]
        await asyncio.gather(*tasks)
    
    async def _generate_description(self, structure: list[TreeNode]) -> str:
        """Generate document description from structure"""
        
        def structure_to_text(nodes: list[TreeNode], indent: int = 0) -> str:
            lines = []
            for node in nodes:
                prefix = "  " * indent
                summary_part = f" - {node.summary[:100]}..." if node.summary else ""
                lines.append(f"{prefix}- {node.title}{summary_part}")
                if node.children:
                    lines.append(structure_to_text(node.children, indent + 1))
            return "\n".join(lines)
        
        structure_text = structure_to_text(structure)
        
        prompt = f'''Based on this document structure, write a one-sentence description
that captures what this document is about. Make it specific enough to
distinguish from similar documents.

Document Structure:
{structure_text}

One-sentence description:'''
        
        return await self.llm.complete(prompt)
```

### 6. Tree Searcher (`searcher.py`)

Reasoning-based retrieval over the tree index.

```python
from dataclasses import dataclass
from typing import Optional
from .models import DocumentIndex, TreeNode
from .llm_client import LLMClient, LLMConfig

@dataclass
class SearchResult:
    """Result from tree search"""
    node_id: str
    title: str
    relevance_score: float
    reasoning: str
    text: str

@dataclass
class SearchConfig:
    """Configuration for tree search"""
    max_iterations: int = 5
    max_results: int = 3
    include_text: bool = True

class TreeSearcher:
    """Reasoning-based search over document tree"""
    
    def __init__(self, llm_config: LLMConfig = None):
        self.llm = LLMClient(llm_config or LLMConfig())
    
    async def search(
        self,
        doc_index: DocumentIndex,
        query: str,
        config: SearchConfig = None,
    ) -> list[SearchResult]:
        """Search document tree for relevant sections"""
        config = config or SearchConfig()
        
        results = []
        visited = set()
        
        for _ in range(config.max_iterations):
            # Get current structure view
            structure_view = self._get_structure_view(doc_index.structure, visited)
            
            if not structure_view:
                break
            
            # Ask LLM to select relevant nodes
            selection = await self._select_nodes(
                query=query,
                structure=structure_view,
                doc_description=doc_index.description,
                previous_results=results,
            )
            
            if not selection.get('node_ids'):
                break
            
            # Process selected nodes
            for node_id in selection['node_ids']:
                if node_id in visited:
                    continue
                visited.add(node_id)
                
                node = self._find_node(node_id, doc_index.structure)
                if node:
                    text = doc_index.get_node_text(node_id) if config.include_text else ""
                    
                    results.append(SearchResult(
                        node_id=node_id,
                        title=node.title,
                        relevance_score=selection.get('scores', {}).get(node_id, 0.8),
                        reasoning=selection.get('reasoning', ''),
                        text=text[:5000] if text else "",  # Truncate for context
                    ))
            
            # Check if LLM thinks we have enough information
            if selection.get('sufficient', False):
                break
            
            if len(results) >= config.max_results:
                break
        
        return results[:config.max_results]
    
    def _get_structure_view(
        self,
        nodes: list[TreeNode],
        visited: set[str],
        max_depth: int = 3,
    ) -> str:
        """Get text representation of structure for LLM"""
        
        def node_to_text(node: TreeNode, depth: int = 0) -> str:
            if depth > max_depth:
                return ""
            
            indent = "  " * depth
            status = "[visited]" if node.node_id in visited else ""
            summary = f" - {node.summary}" if node.summary else ""
            
            lines = [f"{indent}[{node.node_id}] {node.title}{summary} {status}"]
            
            for child in node.children:
                child_text = node_to_text(child, depth + 1)
                if child_text:
                    lines.append(child_text)
            
            return "\n".join(lines)
        
        return "\n".join(node_to_text(n) for n in nodes)
    
    async def _select_nodes(
        self,
        query: str,
        structure: str,
        doc_description: str,
        previous_results: list[SearchResult],
    ) -> dict:
        """Use LLM to select relevant nodes"""
        
        prev_context = ""
        if previous_results:
            prev_context = "Previously found relevant sections:\n"
            for r in previous_results:
                prev_context += f"- [{r.node_id}] {r.title}: {r.text[:200]}...\n"
        
        prompt = f'''You are searching a financial document to answer a query.

Document: {doc_description}

Query: {query}

{prev_context}

Document Structure (node_id in brackets, [visited] = already retrieved):
{structure}

Based on the query, select the most relevant node(s) to retrieve.
Think about which sections would contain the information needed.

Return JSON:
{{
  "reasoning": "why these sections are relevant",
  "node_ids": ["0001", "0002"],
  "scores": {{"0001": 0.9, "0002": 0.7}},
  "sufficient": false  // true if previous results already answer the query
}}'''
        
        return await self.llm.complete_json(prompt)
    
    def _find_node(self, node_id: str, nodes: list[TreeNode]) -> Optional[TreeNode]:
        """Find node by ID in tree"""
        for node in nodes:
            if node.node_id == node_id:
                return node
            if node.children:
                found = self._find_node(node_id, node.children)
                if found:
                    return found
        return None
```

### 7. Main API (`__init__.py`)

Clean public API for the library.

```python
"""
DocumentIndex: Lightweight Hierarchical Tree Index for Financial Documents
"""

from .models import DocumentIndex, DocumentType, TreeNode
from .indexer import DocumentIndexer, IndexerConfig
from .searcher import TreeSearcher, SearchResult, SearchConfig
from .chunker import TextChunker, ChunkerConfig
from .detector import FinancialDocDetector
from .llm_client import LLMClient, LLMConfig

__version__ = "0.1.0"

__all__ = [
    # Main classes
    "DocumentIndexer",
    "TreeSearcher",
    # Models
    "DocumentIndex",
    "DocumentType",
    "TreeNode",
    "SearchResult",
    # Configuration
    "IndexerConfig",
    "SearchConfig",
    "ChunkerConfig",
    "LLMConfig",
    # Utilities
    "TextChunker",
    "FinancialDocDetector",
    "LLMClient",
]


# Convenience functions
async def index_document(
    text: str,
    doc_name: str = "document",
    doc_type: DocumentType = None,
    model: str = "gpt-4o",
    generate_summaries: bool = True,
) -> DocumentIndex:
    """Quick function to index a document"""
    config = IndexerConfig(
        llm_config=LLMConfig(model=model),
        generate_summaries=generate_summaries,
    )
    indexer = DocumentIndexer(config)
    return await indexer.index(text, doc_name, doc_type)


async def search_document(
    doc_index: DocumentIndex,
    query: str,
    model: str = "gpt-4o",
    max_results: int = 3,
) -> list[SearchResult]:
    """Quick function to search a document"""
    searcher = TreeSearcher(LLMConfig(model=model))
    config = SearchConfig(max_results=max_results)
    return await searcher.search(doc_index, query, config)
```

## Usage Examples

### Basic Usage

```python
import asyncio
from documentindex import index_document, search_document, DocumentType

# Example: Index an SEC 10-K filing from edgartools
async def main():
    # Get text from edgartools
    from edgar import Company
    company = Company("AAPL")
    filing = company.get_filings(form="10-K").latest()
    text = filing.text()  # Get full text
    
    # Build hierarchical index
    doc_index = await index_document(
        text=text,
        doc_name=f"AAPL_10K_{filing.filing_date}",
        doc_type=DocumentType.SEC_10K,
    )
    
    # Save index for later use
    import json
    with open("aapl_10k_index.json", "w") as f:
        json.dump(doc_index.to_dict(), f, indent=2)
    
    # Search the document
    results = await search_document(
        doc_index=doc_index,
        query="What are the main risk factors related to supply chain?",
    )
    
    for result in results:
        print(f"[{result.node_id}] {result.title}")
        print(f"Relevance: {result.relevance_score}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Content: {result.text[:500]}...")
        print("---")

asyncio.run(main())
```

### With Earnings Call Transcript

```python
async def index_earnings_call(transcript: str, company: str, quarter: str):
    doc_index = await index_document(
        text=transcript,
        doc_name=f"{company}_{quarter}_earnings",
        doc_type=DocumentType.EARNINGS_CALL,
    )
    
    # Find forward guidance
    results = await search_document(
        doc_index,
        "What guidance did management provide for next quarter?",
    )
    return results
```

### Batch Processing

```python
async def index_multiple_documents(documents: list[tuple[str, str]]):
    """Index multiple documents concurrently"""
    from documentindex import DocumentIndexer, IndexerConfig
    
    indexer = DocumentIndexer(IndexerConfig(
        max_concurrent_requests=10,
        generate_summaries=True,
    ))
    
    tasks = [
        indexer.index(text, name)
        for text, name in documents
    ]
    
    return await asyncio.gather(*tasks)
```

## Project Structure

```
documentindex/
├── __init__.py          # Public API
├── models.py            # Data models (TreeNode, DocumentIndex)
├── chunker.py           # Text chunking with semantic boundaries
├── detector.py          # Financial document type detection
├── llm_client.py        # Multi-provider LLM client
├── indexer.py           # Tree structure builder
├── searcher.py          # Reasoning-based tree search
└── utils.py             # Utility functions
```

## Dependencies

```toml
[project]
dependencies = [
    "litellm>=1.0.0",     # Multi-provider LLM support
    "tiktoken>=0.5.0",    # Token counting
    "pydantic>=2.0.0",    # Data validation (optional)
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]
```

## Comparison with PageIndex

| Feature | PageIndex | DocumentIndex |
|---------|-----------|---------------|
| **Input** | PDF files | Plain text |
| **PDF Parsing** | Built-in (PyPDF2, PyMuPDF) | Not needed |
| **Vision/OCR** | Supported | Not supported |
| **Dependencies** | Heavy (~10+ packages) | Light (~3 packages) |
| **Financial Optimizations** | Generic | SEC filing patterns, earnings calls |
| **LLM Provider** | OpenAI only | Multiple via litellm |
| **Async** | Partial | Native async throughout |
| **Tree Search** | Supported | Supported |
| **Document Type Detection** | Basic | Financial-focused |

## Future Enhancements

1. **Caching**: Add Redis/file-based caching for index persistence
2. **Streaming**: Support streaming for large document processing
3. **Metadata Extraction**: Extract entities, dates, numbers from financial docs
4. **Cross-Reference Resolution**: Follow "see Appendix G" style references
5. **Multi-Document Index**: Index across multiple related documents
6. **Vector Hybrid**: Optional vector similarity as fallback

## Summary

DocumentIndex provides a lightweight, text-focused alternative to PageIndex specifically designed for financial documents. By eliminating PDF parsing complexity and focusing on text input, it offers:

- **Simpler Integration**: Works with any text source (edgartools, web scrapers, document converters)
- **Faster Processing**: No PDF/vision overhead
- **Financial Focus**: Built-in patterns for SEC filings, earnings calls, research reports
- **Modern Async**: Native async support for concurrent processing
- **LLM Flexibility**: Support for multiple LLM providers

The reasoning-based tree search approach maintains the key advantage of PageIndex (semantic relevance over similarity) while being more practical for financial document workflows.
