"""
Data models for DocumentIndex.

Provides:
- Document types for financial documents
- Tree structure models (TreeNode, TextSpan)
- Document index container
- Search and retrieval result models
"""

from dataclasses import dataclass, field
from typing import Optional, Any, TYPE_CHECKING
from enum import Enum
from datetime import datetime
import json

if TYPE_CHECKING:
    from typing import Self


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
    key_numbers: dict[str, Any] = field(default_factory=dict)
    key_dates: list[tuple[str, str]] = field(default_factory=list)  # (description, date_str)
    
    # Custom metadata
    custom: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "company_name": self.company_name,
            "ticker": self.ticker,
            "cik": self.cik,
            "filing_date": self.filing_date.isoformat() if self.filing_date else None,
            "period_end_date": self.period_end_date.isoformat() if self.period_end_date else None,
            "fiscal_year": self.fiscal_year,
            "fiscal_quarter": self.fiscal_quarter,
            "key_people": self.key_people,
            "key_numbers": self.key_numbers,
            "key_dates": self.key_dates,
            "custom": self.custom,
        }


@dataclass
class CrossReference:
    """A reference from one part of document to another"""
    source_node_id: str
    target_description: str  # e.g., "Appendix G", "Note 15", "Table 5.3"
    target_node_id: Optional[str] = None  # Resolved target
    reference_text: str = ""  # Original text containing the reference
    resolved: bool = False
    
    def to_dict(self) -> dict:
        return {
            "source_node_id": self.source_node_id,
            "target_description": self.target_description,
            "target_node_id": self.target_node_id,
            "reference_text": self.reference_text[:200] if self.reference_text else "",
            "resolved": self.resolved,
        }


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
        # Allow equal for empty spans
        if self.end_char < self.start_char:
            raise ValueError(f"end_char ({self.end_char}) must be >= start_char ({self.start_char})")
        if self.end_chunk < self.start_chunk:
            raise ValueError(f"end_chunk ({self.end_chunk}) must be >= start_chunk ({self.start_chunk})")
    
    def to_dict(self) -> dict:
        return {
            "start_char": self.start_char,
            "end_char": self.end_char,
            "start_chunk": self.start_chunk,
            "end_chunk": self.end_chunk,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TextSpan":
        return cls(
            start_char=data["start_char"],
            end_char=data["end_char"],
            start_chunk=data["start_chunk"],
            end_chunk=data["end_chunk"],
        )


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
    children: list["TreeNode"] = field(default_factory=list)
    
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
    
    @property
    def start_char(self) -> int:
        """Character start offset"""
        return self.text_span.start_char
    
    @property
    def end_char(self) -> int:
        """Character end offset"""
        return self.text_span.end_char
    
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
            result["cross_references"] = [cr.to_dict() for cr in self.cross_references]
        if include_children and self.children:
            result["nodes"] = [child.to_dict() for child in self.children]
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> "TreeNode":
        """Create TreeNode from dictionary"""
        text_span = TextSpan(
            start_char=data.get("start_char", 0),
            end_char=data.get("end_char", 0),
            start_chunk=data.get("start_index", 0),
            end_chunk=data.get("end_index", 0),
        )
        
        node = cls(
            node_id=data["node_id"],
            title=data["title"],
            level=data.get("level", 0),
            text_span=text_span,
            summary=data.get("summary"),
            parent_id=data.get("parent_id"),
            node_metadata=data.get("metadata", {}),
        )
        
        # Parse children recursively
        if "nodes" in data:
            node.children = [cls.from_dict(child) for child in data["nodes"]]
        
        # Parse cross-references
        if "cross_references" in data:
            node.cross_references = [
                CrossReference(
                    source_node_id=node.node_id,
                    target_description=cr.get("target_description", cr.get("target", "")),
                    target_node_id=cr.get("target_node_id", cr.get("resolved_to")),
                    resolved=cr.get("resolved", cr.get("target_node_id") is not None),
                )
                for cr in data["cross_references"]
            ]
        
        return node


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
        nodes: list[TreeNode] = []
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
        path: list[TreeNode] = []
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
    
    def get_node_count(self) -> int:
        """Get total number of nodes"""
        return len(self.get_all_nodes())
    
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
            "metadata": self.metadata.to_dict(),
            "structure": [node.to_dict() for node in self.structure],
        }
        if include_text:
            result["original_text"] = self.original_text
            result["chunks"] = self.chunks
            result["chunk_char_offsets"] = self.chunk_char_offsets
        return result
    
    def to_json(self, include_text: bool = False, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(include_text), indent=indent, ensure_ascii=False)
    
    def save(self, path: str, include_text: bool = True):
        """Save index to JSON file"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json(include_text))
    
    @classmethod
    def from_dict(cls, data: dict, original_text: str = None) -> "DocumentIndex":
        """Reconstruct from dictionary"""
        # Parse metadata
        metadata_data = data.get("metadata", {})
        metadata = DocumentMetadata(
            company_name=metadata_data.get("company_name"),
            ticker=metadata_data.get("ticker"),
            cik=metadata_data.get("cik"),
            fiscal_year=metadata_data.get("fiscal_year"),
            fiscal_quarter=metadata_data.get("fiscal_quarter"),
            key_people=metadata_data.get("key_people", []),
            key_numbers=metadata_data.get("key_numbers", {}),
            key_dates=metadata_data.get("key_dates", []),
            custom=metadata_data.get("custom", {}),
        )
        
        # Parse dates
        if metadata_data.get("filing_date"):
            try:
                metadata.filing_date = datetime.fromisoformat(metadata_data["filing_date"])
            except (ValueError, TypeError):
                pass
        if metadata_data.get("period_end_date"):
            try:
                metadata.period_end_date = datetime.fromisoformat(metadata_data["period_end_date"])
            except (ValueError, TypeError):
                pass
        
        # Parse structure
        structure = [TreeNode.from_dict(node) for node in data.get("structure", [])]
        
        # Get text data
        text = original_text or data.get("original_text", "")
        chunks = data.get("chunks", [])
        chunk_offsets = [tuple(o) for o in data.get("chunk_char_offsets", [])]
        
        # Parse created_at
        created_at = datetime.now()
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"])
            except (ValueError, TypeError):
                pass
        
        return cls(
            doc_id=data["doc_id"],
            doc_name=data["doc_name"],
            doc_type=DocumentType(data["doc_type"]),
            original_text=text,
            chunks=chunks,
            chunk_char_offsets=chunk_offsets,
            structure=structure,
            metadata=metadata,
            description=data.get("description"),
            created_at=created_at,
            index_version=data.get("index_version", "1.0"),
        )
    
    @classmethod
    def load(cls, path: str) -> "DocumentIndex":
        """Load index from JSON file"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


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
        return doc_index.get_node_text(self.node.node_id) or ""
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node.node_id,
            "title": self.node.title,
            "relevance_score": self.relevance_score,
            "match_reason": self.match_reason,
            "matched_excerpts": self.matched_excerpts,
            "start_char": self.node.start_char,
            "end_char": self.node.end_char,
        }


@dataclass
class Citation:
    """A citation to a specific part of the document"""
    node_id: str
    node_title: str
    excerpt: str  # The specific text being cited
    char_range: tuple[int, int]  # Position in original text
    
    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_title": self.node_title,
            "excerpt": self.excerpt,
            "char_range": list(self.char_range),
        }


@dataclass
class QAResult:
    """Result from agentic question answering"""
    question: str
    answer: str
    confidence: float
    citations: list[Citation]
    reasoning_trace: list[dict]  # Steps taken to find answer
    nodes_visited: list[str]  # Node IDs visited during search
    
    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "confidence": self.confidence,
            "citations": [c.to_dict() for c in self.citations],
            "reasoning_trace": self.reasoning_trace,
            "nodes_visited": self.nodes_visited,
        }


@dataclass
class ProvenanceResult:
    """Result from provenance extraction"""
    topic: str
    evidence: list[NodeMatch]
    total_nodes_scanned: int
    scan_coverage: float  # Percentage of document scanned
    summary: Optional[str] = None  # Optional summary of all evidence
    
    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "evidence": [e.to_dict() for e in self.evidence],
            "total_nodes_scanned": self.total_nodes_scanned,
            "scan_coverage": self.scan_coverage,
            "summary": self.summary,
        }
