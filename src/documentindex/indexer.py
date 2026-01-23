"""
Document indexer - builds hierarchical tree structure from text.

The indexer:
1. Chunks the document text
2. Detects document type
3. Uses LLM to identify structure
4. Builds hierarchical tree with text mapping
5. Optionally generates summaries
6. Extracts metadata
7. Resolves cross-references
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import asyncio
import uuid
import logging
import re

from .models import (
    DocumentIndex, DocumentType, TreeNode, TextSpan,
    DocumentMetadata,
)
from .chunker import TextChunker, ChunkConfig, Chunk
from .detector import FinancialDocDetector
from .llm_client import LLMClient, LLMConfig
from .metadata import MetadataExtractor, MetadataExtractorConfig
from .cross_ref import CrossReferenceResolver, CrossRefConfig
from .cache import CacheManager
from .streaming import (
    ProgressCallback, ProgressUpdate, OperationType, ProgressReporter
)

logger = logging.getLogger(__name__)


@dataclass
class IndexerConfig:
    """Configuration for document indexer"""
    # LLM settings
    llm_config: Optional[LLMConfig] = None
    
    # Chunking settings
    chunk_config: Optional[ChunkConfig] = None
    
    # Feature flags
    generate_summaries: bool = True
    extract_metadata: bool = True
    resolve_cross_refs: bool = True
    
    # Processing settings
    max_concurrent_summaries: int = 5
    large_section_threshold: int = 10  # Chunks - sections larger than this get recursively processed
    
    # Cache
    use_cache: bool = True


class DocumentIndexer:
    """
    Builds hierarchical tree index from document text.
    
    The indexer uses LLM to understand document structure and
    build a navigable tree that maps to the original text.
    """
    
    def __init__(
        self,
        config: Optional[IndexerConfig] = None,
        llm_client: Optional[LLMClient] = None,
        cache_manager: Optional[CacheManager] = None,
    ):
        self.config = config or IndexerConfig()
        self.llm = llm_client or LLMClient(self.config.llm_config or LLMConfig())
        self.chunker = TextChunker(self.config.chunk_config)
        self.cache = cache_manager
        
        # Sub-components
        self.metadata_extractor = MetadataExtractor(MetadataExtractorConfig(
            use_llm_extraction=False,  # Use sync extraction for speed
        ))
        self.cross_ref_resolver = CrossReferenceResolver(CrossRefConfig(
            use_llm_resolution=False,  # Use pattern matching only
        ))
    
    async def index(
        self,
        text: str,
        doc_name: str = "document",
        doc_type: Optional[DocumentType] = None,
        doc_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> DocumentIndex:
        """
        Build index for a document.
        
        Args:
            text: Full document text
            doc_name: Name for the document
            doc_type: Document type (auto-detected if not provided)
            doc_id: Unique ID (generated if not provided)
            description: Optional description
        
        Returns:
            DocumentIndex with hierarchical structure
        """
        return await self.index_with_progress(
            text=text,
            doc_name=doc_name,
            doc_type=doc_type,
            doc_id=doc_id,
            description=description,
            progress_callback=None,
        )
    
    async def index_with_progress(
        self,
        text: str,
        doc_name: str = "document",
        doc_type: Optional[DocumentType] = None,
        doc_id: Optional[str] = None,
        description: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> DocumentIndex:
        """
        Build index with progress reporting.
        
        Progress steps:
        1. Chunking text
        2. Detecting document type
        3. Building structure
        4. Generating summaries
        5. Extracting metadata
        6. Resolving cross-references
        7. Finalizing index
        """
        reporter = ProgressReporter(
            operation=OperationType.INDEXING,
            total_steps=7,
            callback=progress_callback,
        )
        
        # Generate doc_id if not provided
        if not doc_id:
            doc_id = str(uuid.uuid4())[:8]
        
        # Step 1: Chunk text
        reporter.report("Chunking", f"Processing {len(text)} characters")
        chunks = self.chunker.chunk(text)
        chunk_texts = [c.text for c in chunks]
        chunk_offsets = [(c.start_char, c.end_char) for c in chunks]
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 2: Detect document type
        reporter.report("Type Detection", "Analyzing document structure")
        if doc_type is None:
            doc_type = FinancialDocDetector.detect(text, doc_name)
        
        logger.info(f"Detected document type: {doc_type.value}")
        
        # Step 3: Build structure
        reporter.report("Structure Building", f"Analyzing {len(chunks)} chunks")
        structure = await self._build_structure(chunks, doc_type)
        
        logger.info(f"Built structure with {len(structure)} root nodes")
        
        # Step 4: Generate summaries
        if self.config.generate_summaries and structure:
            reporter.report("Summary Generation", f"Summarizing sections")
            await self._generate_summaries(structure, chunk_texts)
        else:
            reporter.report("Summary Generation", "Skipped")
        
        # Step 5: Extract metadata
        if self.config.extract_metadata:
            reporter.report("Metadata Extraction", "Extracting document metadata")
            metadata = self.metadata_extractor.extract_sync(text, doc_type, doc_name)
        else:
            reporter.report("Metadata Extraction", "Skipped")
            metadata = DocumentMetadata()
        
        # Step 6: Create document index (needed for cross-ref resolution)
        reporter.report("Building Index", "Creating index structure")
        doc_index = DocumentIndex(
            doc_id=doc_id,
            doc_name=doc_name,
            doc_type=doc_type,
            original_text=text,
            chunks=chunk_texts,
            chunk_char_offsets=chunk_offsets,
            structure=structure,
            metadata=metadata,
            description=description or f"{doc_type.value} document: {doc_name}",
        )
        
        # Step 7: Resolve cross-references
        if self.config.resolve_cross_refs:
            reporter.report("Cross-Reference Resolution", "Finding and resolving references")
            self.cross_ref_resolver.resolve_references_sync(doc_index)
        else:
            reporter.report("Cross-Reference Resolution", "Skipped")
        
        logger.info(f"Indexing complete: {doc_index.get_node_count()} nodes")
        
        return doc_index
    
    async def _build_structure(
        self,
        chunks: list[Chunk],
        doc_type: DocumentType,
    ) -> list[TreeNode]:
        """Build hierarchical structure using LLM"""
        if not chunks:
            return []
        
        # Build chunk summary for structure detection
        chunks_summary = self._build_chunks_summary(chunks)
        
        # Get structure from LLM
        prompt = f"""Analyze this document's structure and identify the hierarchical sections.

Document type: {doc_type.value}

The document has been split into chunks. Here are the chunk markers and their starting content:

{chunks_summary}

Identify the hierarchical structure of the document. For each section, provide:
- structure: A hierarchical number like "1", "1.1", "1.1.1"
- title: The section title
- chunk_index: The starting chunk index (0-based)
- end_chunk_index: The ending chunk index (exclusive)

Return a JSON array:
[
  {{"structure": "1", "title": "PART I - FINANCIAL INFORMATION", "chunk_index": 0, "end_chunk_index": 5}},
  {{"structure": "1.1", "title": "Item 1. Financial Statements", "chunk_index": 1, "end_chunk_index": 3}},
  {{"structure": "1.2", "title": "Item 2. MD&A", "chunk_index": 3, "end_chunk_index": 5}},
  {{"structure": "2", "title": "PART II - OTHER INFORMATION", "chunk_index": 5, "end_chunk_index": 8}}
]

Focus on major sections. For SEC filings, look for PART, ITEM, and major subsections.
For earnings calls, look for Presentation, Q&A, and speaker sections.

Return valid JSON array only."""

        try:
            result = await self.llm.complete_json(prompt)
            
            if isinstance(result, list):
                # Convert to TreeNodes
                return self._build_tree_from_flat(result, chunks)
            else:
                logger.warning("LLM returned non-list structure, using fallback")
                return self._build_fallback_structure(chunks)
        except Exception as e:
            logger.warning(f"Structure detection failed: {e}, using fallback")
            return self._build_fallback_structure(chunks)
    
    def _build_chunks_summary(self, chunks: list[Chunk]) -> str:
        """Build summary of chunks for LLM"""
        lines = []
        for i, chunk in enumerate(chunks[:50]):  # Limit chunks shown
            # Get first 200 chars of chunk, clean up whitespace
            preview = chunk.text[:200].replace('\n', ' ').strip()
            section_info = f" [{chunk.section_title}]" if chunk.section_title else ""
            lines.append(f"<chunk_{i}>{section_info}: {preview}...")
        
        if len(chunks) > 50:
            lines.append(f"... and {len(chunks) - 50} more chunks")
        
        return "\n".join(lines)
    
    def _build_tree_from_flat(
        self,
        flat_structure: list[dict],
        chunks: list[Chunk],
    ) -> list[TreeNode]:
        """Convert flat structure list to hierarchical tree"""
        if not flat_structure:
            return self._build_fallback_structure(chunks)
        
        # Validate and fix indices
        max_chunk = len(chunks)
        for item in flat_structure:
            item["chunk_index"] = max(0, min(item.get("chunk_index", 0), max_chunk - 1))
            item["end_chunk_index"] = max(
                item["chunk_index"] + 1,
                min(item.get("end_chunk_index", max_chunk), max_chunk)
            )
        
        # Sort by structure number
        flat_structure.sort(key=lambda x: self._structure_sort_key(x.get("structure", "0")))
        
        # Build nodes
        nodes_by_structure: dict[str, TreeNode] = {}
        root_nodes: list[TreeNode] = []
        
        for i, item in enumerate(flat_structure):
            structure = item.get("structure", str(i))
            title = item.get("title", f"Section {i}")
            start_chunk = item["chunk_index"]
            end_chunk = item["end_chunk_index"]
            
            # Calculate char offsets from chunks
            start_char = chunks[start_chunk].start_char if start_chunk < len(chunks) else 0
            end_char = chunks[min(end_chunk, len(chunks)) - 1].end_char if end_chunk > 0 else start_char
            
            # Determine level from structure
            level = len(structure.split(".")) - 1
            
            # Generate node_id
            node_id = f"{i:04d}"
            
            node = TreeNode(
                node_id=node_id,
                title=title,
                level=level,
                text_span=TextSpan(
                    start_char=start_char,
                    end_char=end_char,
                    start_chunk=start_chunk,
                    end_chunk=end_chunk,
                ),
            )
            
            nodes_by_structure[structure] = node
            
            # Find parent
            parent_structure = self._get_parent_structure(structure)
            if parent_structure and parent_structure in nodes_by_structure:
                parent = nodes_by_structure[parent_structure]
                node.parent_id = parent.node_id
                parent.children.append(node)
            else:
                root_nodes.append(node)
        
        return root_nodes
    
    def _structure_sort_key(self, structure: str) -> tuple:
        """Generate sort key for structure numbers"""
        parts = structure.split(".")
        result = []
        for part in parts:
            try:
                result.append(int(part))
            except ValueError:
                # Handle roman numerals or letters
                result.append(self._roman_to_int(part) if part.isalpha() else 0)
        return tuple(result)
    
    def _roman_to_int(self, s: str) -> int:
        """Convert Roman numeral to integer"""
        roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        s = s.upper()
        result = 0
        prev = 0
        for char in reversed(s):
            curr = roman_values.get(char, 0)
            if curr < prev:
                result -= curr
            else:
                result += curr
            prev = curr
        return result
    
    def _get_parent_structure(self, structure: str) -> Optional[str]:
        """Get parent structure number"""
        parts = structure.split(".")
        if len(parts) > 1:
            return ".".join(parts[:-1])
        return None
    
    def _build_fallback_structure(self, chunks: list[Chunk]) -> list[TreeNode]:
        """Build simple structure when LLM fails"""
        if not chunks:
            return []
        
        # Group chunks by detected sections
        sections: list[tuple[str, int, int]] = []  # (title, start, end)
        current_section = None
        current_start = 0
        
        for i, chunk in enumerate(chunks):
            if chunk.section_title and chunk.section_title != current_section:
                if current_section is not None:
                    sections.append((current_section, current_start, i))
                current_section = chunk.section_title
                current_start = i
        
        # Add final section
        if current_section is not None:
            sections.append((current_section, current_start, len(chunks)))
        
        # If no sections detected, create one root node
        if not sections:
            return [TreeNode(
                node_id="0000",
                title="Document",
                level=0,
                text_span=TextSpan(
                    start_char=chunks[0].start_char,
                    end_char=chunks[-1].end_char,
                    start_chunk=0,
                    end_chunk=len(chunks),
                ),
            )]
        
        # Create nodes
        nodes = []
        for i, (title, start, end) in enumerate(sections):
            nodes.append(TreeNode(
                node_id=f"{i:04d}",
                title=title,
                level=0,
                text_span=TextSpan(
                    start_char=chunks[start].start_char,
                    end_char=chunks[min(end, len(chunks)) - 1].end_char,
                    start_chunk=start,
                    end_chunk=end,
                ),
            ))
        
        return nodes
    
    async def _generate_summaries(
        self,
        nodes: list[TreeNode],
        chunks: list[str],
    ) -> None:
        """Generate summaries for all nodes"""
        all_nodes = []
        self._collect_all_nodes(nodes, all_nodes)
        
        # Generate summaries concurrently with limit
        semaphore = asyncio.Semaphore(self.config.max_concurrent_summaries)
        
        async def summarize_node(node: TreeNode):
            async with semaphore:
                text = "\n".join(chunks[node.start_index:node.end_index])
                if len(text) > 100:  # Only summarize substantial content
                    node.summary = await self._generate_summary(text, node.title)
        
        await asyncio.gather(*[summarize_node(n) for n in all_nodes])
    
    def _collect_all_nodes(self, nodes: list[TreeNode], result: list[TreeNode]):
        """Collect all nodes recursively"""
        for node in nodes:
            result.append(node)
            if node.children:
                self._collect_all_nodes(node.children, result)
    
    async def _generate_summary(self, text: str, title: str) -> str:
        """Generate summary for a section"""
        # Truncate text if too long
        if len(text) > 6000:
            text = text[:3000] + "\n...[truncated]...\n" + text[-3000:]
        
        prompt = f"""Summarize this document section in 2-3 sentences.

Section: {title}

Content:
{text}

Provide a concise summary focusing on the key information:"""

        try:
            summary = await self.llm.complete(prompt)
            return summary.strip()[:500]  # Limit summary length
        except Exception as e:
            logger.warning(f"Summary generation failed for {title}: {e}")
            return ""


# ============================================================================
# Convenience functions
# ============================================================================

async def index_document(
    text: str,
    doc_name: str = "document",
    doc_type: Optional[DocumentType] = None,
    model: str = "gpt-4o",
) -> DocumentIndex:
    """
    Convenience function to index a document.
    
    Args:
        text: Document text
        doc_name: Document name
        doc_type: Document type (auto-detected if not provided)
        model: LLM model to use
    
    Returns:
        DocumentIndex
    """
    config = IndexerConfig(
        llm_config=LLMConfig(model=model),
    )
    indexer = DocumentIndexer(config)
    return await indexer.index(text, doc_name, doc_type)
