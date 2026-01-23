"""
DocumentIndex: Lightweight hierarchical tree index for financial documents.

Provides:
- Document indexing with hierarchical tree structure
- Two retrieval modes:
  - Agentic QA: Intelligent, iterative question answering
  - Provenance Extraction: Exhaustive evidence gathering
- Multi-provider LLM support (OpenAI, Anthropic, AWS Bedrock, Azure)
- Streaming responses and progress tracking
- Caching for improved performance

Quick Start:
    from documentindex import DocumentIndexer, AgenticQA, ProvenanceExtractor

    # Index a document
    indexer = DocumentIndexer()
    doc_index = await indexer.index(text, doc_name="10K_2024")

    # Answer questions
    qa = AgenticQA(doc_index)
    result = await qa.answer("What was the revenue?")

    # Extract all evidence about a topic
    extractor = ProvenanceExtractor(doc_index)
    evidence = await extractor.extract_all("climate change risks")
"""

__version__ = "0.1.0"

# ============================================================================
# Core Data Models
# ============================================================================

from .models import (
    # Document types
    DocumentType,
    
    # Structure models
    TextSpan,
    TreeNode,
    DocumentIndex,
    
    # Metadata
    DocumentMetadata,
    CrossReference,
    
    # Search/Retrieval results
    NodeMatch,
    Citation,
    QAResult,
    ProvenanceResult,
)

# ============================================================================
# LLM Client
# ============================================================================

from .llm_client import (
    LLMConfig,
    LLMClient,
    
    # Factory functions
    create_openai_client,
    create_anthropic_client,
    create_bedrock_client,
    create_azure_client,
    create_ollama_client,
)

# ============================================================================
# Document Processing
# ============================================================================

from .chunker import (
    ChunkConfig,
    Chunk,
    TextChunker,
    count_tokens,
)

from .detector import (
    FinancialDocDetector,
)

from .indexer import (
    IndexerConfig,
    DocumentIndexer,
    index_document,
)

from .metadata import (
    MetadataExtractorConfig,
    MetadataExtractor,
)

from .cross_ref import (
    CrossRefConfig,
    CrossReferenceDetector,
    CrossReferenceResolver,
    CrossReferenceFollower,
)

# ============================================================================
# Retrieval Components
# ============================================================================

from .searcher import (
    NodeSearchConfig,
    NodeSearcher,
    search_nodes,
)

from .agentic_qa import (
    AgenticQAConfig,
    AgenticQA,
    answer_question,
)

from .provenance import (
    ProvenanceConfig,
    ProvenanceExtractor,
    extract_provenance,
    extract_multiple_topics,
)

# ============================================================================
# Streaming & Progress
# ============================================================================

from .streaming import (
    OperationType,
    ProgressUpdate,
    ProgressCallback,
    StreamChunk,
    StreamingQAResult,
    StreamingProvenanceResult,
    ProgressReporter,
)

# ============================================================================
# Caching
# ============================================================================

from .cache import (
    CacheConfig,
    CacheManager,
    MemoryCache,
    FileCache,
    RedisCache,
)

# ============================================================================
# Utilities
# ============================================================================

from .utils import (
    generate_doc_id,
    truncate_text,
    truncate_middle,
    clean_text,
    extract_sentences,
    parse_date,
    estimate_tokens,
    chunk_text_simple,
)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    # Version
    "__version__",
    
    # Core models
    "DocumentType",
    "TextSpan",
    "TreeNode",
    "DocumentIndex",
    "DocumentMetadata",
    "CrossReference",
    "NodeMatch",
    "Citation",
    "QAResult",
    "ProvenanceResult",
    
    # LLM
    "LLMConfig",
    "LLMClient",
    "create_openai_client",
    "create_anthropic_client",
    "create_bedrock_client",
    "create_azure_client",
    "create_ollama_client",
    
    # Chunking
    "ChunkConfig",
    "Chunk",
    "TextChunker",
    "count_tokens",
    
    # Detection
    "FinancialDocDetector",
    
    # Indexing
    "IndexerConfig",
    "DocumentIndexer",
    "index_document",
    
    # Metadata
    "MetadataExtractorConfig",
    "MetadataExtractor",
    
    # Cross-references
    "CrossRefConfig",
    "CrossReferenceDetector",
    "CrossReferenceResolver",
    "CrossReferenceFollower",
    
    # Search
    "NodeSearchConfig",
    "NodeSearcher",
    "search_nodes",
    
    # QA
    "AgenticQAConfig",
    "AgenticQA",
    "answer_question",
    
    # Provenance
    "ProvenanceConfig",
    "ProvenanceExtractor",
    "extract_provenance",
    "extract_multiple_topics",
    
    # Streaming
    "OperationType",
    "ProgressUpdate",
    "ProgressCallback",
    "StreamChunk",
    "StreamingQAResult",
    "StreamingProvenanceResult",
    "ProgressReporter",
    
    # Cache
    "CacheConfig",
    "CacheManager",
    "MemoryCache",
    "FileCache",
    "RedisCache",
    
    # Utils
    "generate_doc_id",
    "truncate_text",
    "truncate_middle",
    "clean_text",
    "extract_sentences",
    "parse_date",
    "estimate_tokens",
    "chunk_text_simple",
]
