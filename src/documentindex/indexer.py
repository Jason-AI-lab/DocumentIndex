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
import json
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
    summary_llm_config: Optional[LLMConfig] = None  # Cheaper model for summaries (defaults to llm_config)

    # Chunking settings
    chunk_config: Optional[ChunkConfig] = None

    # Feature flags
    generate_summaries: bool = True
    extract_metadata: bool = True
    resolve_cross_refs: bool = True

    # Processing settings
    max_concurrent_summaries: int = 5
    large_section_threshold: int = 10  # Chunks - sections larger than this get recursively processed

    # Batch summary settings
    summary_batch_size: int = 10  # Nodes per batch LLM call
    max_concurrent_batches: int = 3  # Parallel batch processing limit
    summary_token_budget: int = 8000  # Max input tokens per summary batch

    # Small document optimization
    small_doc_threshold: int = 15  # Chunks - documents smaller than this use combined structure+summary

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
        self.summary_llm = (
            LLMClient(self.config.summary_llm_config)
            if self.config.summary_llm_config
            else self.llm
        )
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
        
        # Step 1: Detect document type (must happen before chunking)
        reporter.report("Type Detection", "Analyzing document structure")
        if doc_type is None:
            doc_type = FinancialDocDetector.detect(text, doc_name)

        logger.info(f"Detected document type: {doc_type.value}")

        # Step 2: Chunk text (with type-specific section patterns)
        reporter.report("Chunking", f"Processing {len(text)} characters")
        self.chunker.set_doc_type(doc_type)
        chunks = self.chunker.chunk(text)
        chunk_texts = [c.text for c in chunks]
        chunk_offsets = [(c.start_char, c.end_char) for c in chunks]

        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 3 & 4: Build structure (and optionally generate summaries in one call)
        summaries_done = False
        if (
            self.config.generate_summaries
            and len(chunks) <= self.config.small_doc_threshold
        ):
            # Small document: combine structure + summary in a single LLM call
            reporter.report(
                "Structure + Summaries",
                f"Analyzing {len(chunks)} chunks (combined)",
            )
            structure = await self._build_structure_with_summaries(chunks, doc_type)
            summaries_done = True
            logger.info(f"Built structure with summaries: {len(structure)} root nodes")
        else:
            reporter.report("Structure Building", f"Analyzing {len(chunks)} chunks")
            structure = await self._build_structure(chunks, doc_type)
            logger.info(f"Built structure with {len(structure)} root nodes")

        # Step 4 & 5: Generate summaries + extract metadata in parallel
        async def _run_summaries():
            if not summaries_done and self.config.generate_summaries and structure:
                reporter.report("Summary Generation", "Summarizing sections")
                await self._generate_summaries(structure, chunk_texts)
            elif not summaries_done:
                reporter.report("Summary Generation", "Skipped")

        async def _run_metadata():
            if self.config.extract_metadata:
                reporter.report("Metadata Extraction", "Extracting document metadata")
                return self.metadata_extractor.extract_sync(text, doc_type, doc_name)
            else:
                reporter.report("Metadata Extraction", "Skipped")
                return DocumentMetadata()

        _, metadata = await asyncio.gather(_run_summaries(), _run_metadata())
        
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
        """Build hierarchical structure, skipping LLM when chunks have clear section metadata"""
        if not chunks:
            return []

        # Check if chunks already have sufficient section metadata from the chunker
        sections_found = [c for c in chunks if c.section_title]
        distinct_sections = len(set(c.section_title for c in sections_found))

        if distinct_sections >= 3 and len(sections_found) / len(chunks) > 0.2:
            logger.info(
                f"Found {distinct_sections} distinct sections in chunk metadata, "
                "building structure from chunks (skipping LLM)"
            )
            return self._build_structure_from_chunks(chunks)

        # Fall through to LLM-based structure detection
        chunks_summary = self._build_chunks_summary(chunks)

        if doc_type == DocumentType.EARNINGS_CALL:
            prompt = self._earnings_call_structure_prompt(chunks_summary, len(chunks))
        else:
            prompt = self._generic_structure_prompt(chunks_summary, doc_type)

        try:
            result = await self._cached_llm_complete_json(prompt)

            if isinstance(result, list):
                return self._build_tree_from_flat(result, chunks)
            else:
                logger.warning("LLM returned non-list structure, using fallback")
                return self._build_fallback_structure(chunks)
        except Exception as e:
            logger.warning(f"Structure detection failed: {e}, using fallback")
            return self._build_fallback_structure(chunks)

    def _build_structure_from_chunks(self, chunks: list[Chunk]) -> list[TreeNode]:
        """Build hierarchical tree directly from chunk section metadata (no LLM needed)"""
        # Group consecutive chunks by section, respecting hierarchy levels
        sections: list[dict] = []
        current_title = None
        current_level = 0
        current_start = 0

        for i, chunk in enumerate(chunks):
            if chunk.section_title and chunk.section_title != current_title:
                if current_title is not None:
                    sections.append({
                        "title": current_title,
                        "level": current_level,
                        "start_chunk": current_start,
                        "end_chunk": i,
                    })
                current_title = chunk.section_title
                current_level = chunk.section_level
                current_start = i
            elif not chunk.section_title and current_title is None:
                # Chunks before any section header
                current_title = "Introduction"
                current_level = 0
                current_start = i

        # Add final section
        if current_title is not None:
            sections.append({
                "title": current_title,
                "level": current_level,
                "start_chunk": current_start,
                "end_chunk": len(chunks),
            })

        if not sections:
            return self._build_fallback_structure(chunks)

        # Build hierarchical tree from flat sections with levels
        root_nodes: list[TreeNode] = []
        stack: list[TreeNode] = []  # Stack of parent nodes

        for i, section in enumerate(sections):
            start_chunk = section["start_chunk"]
            end_chunk = section["end_chunk"]
            level = section["level"]

            node = TreeNode(
                node_id=f"{i:04d}",
                title=section["title"],
                level=level,
                text_span=TextSpan(
                    start_char=chunks[start_chunk].start_char,
                    end_char=chunks[min(end_chunk, len(chunks)) - 1].end_char,
                    start_chunk=start_chunk,
                    end_chunk=end_chunk,
                ),
            )

            # Find appropriate parent by popping stack until we find a node at a lower level
            while stack and stack[-1].level >= level:
                stack.pop()

            if stack:
                parent = stack[-1]
                node.parent_id = parent.node_id
                parent.children.append(node)
                # Expand parent span to include this child
                parent.text_span = TextSpan(
                    start_char=parent.text_span.start_char,
                    end_char=max(parent.text_span.end_char, node.text_span.end_char),
                    start_chunk=parent.text_span.start_chunk,
                    end_chunk=max(parent.text_span.end_chunk, node.text_span.end_chunk),
                )
            else:
                root_nodes.append(node)

            stack.append(node)

        return root_nodes

    async def _build_structure_with_summaries(
        self,
        chunks: list[Chunk],
        doc_type: DocumentType,
    ) -> list[TreeNode]:
        """Build structure AND generate summaries in a single LLM call for small documents."""
        if not chunks:
            return []

        # Include full chunk text (small doc, so this fits in context)
        chunk_content = []
        for i, chunk in enumerate(chunks):
            section_info = f" [{chunk.section_title}]" if chunk.section_title else ""
            preview = chunk.text[:500].replace('\n', ' ').strip()
            chunk_content.append(f"<chunk_{i}>{section_info}: {preview}")

        chunks_text = "\n".join(chunk_content)

        prompt = f"""Analyze this document and provide both its hierarchical structure and a 2-3 sentence summary for each section.

Document type: {doc_type.value}

The document has {len(chunks)} chunks:

{chunks_text}

Return a JSON array where each object has:
- structure: Hierarchical number ("1", "1.1", etc.)
- title: Section title
- chunk_index: Starting chunk index (0-based)
- end_chunk_index: Ending chunk index (exclusive)
- summary: 2-3 sentence summary of this section

Example:
[
  {{"structure": "1", "title": "PART I", "chunk_index": 0, "end_chunk_index": 3, "summary": "Overview of..."}},
  {{"structure": "1.1", "title": "Item 1. Business", "chunk_index": 0, "end_chunk_index": 2, "summary": "Describes..."}}
]

Return valid JSON array only."""

        try:
            result = await self._cached_llm_complete_json(prompt)

            if isinstance(result, list):
                nodes = self._build_tree_from_flat(result, chunks)
                # Apply summaries from the combined response
                nodes_by_id: dict[str, TreeNode] = {}
                all_nodes: list[TreeNode] = []
                self._collect_all_nodes(nodes, all_nodes)
                for n in all_nodes:
                    nodes_by_id[n.node_id] = n

                for i, item in enumerate(result):
                    node_id = f"{i:04d}"
                    if node_id in nodes_by_id and "summary" in item:
                        nodes_by_id[node_id].summary = str(item["summary"])[:500]

                return nodes
            else:
                logger.warning("Combined call returned non-list, falling back")
                return self._build_fallback_structure(chunks)
        except Exception as e:
            logger.warning(f"Combined structure+summary failed: {e}, falling back")
            return self._build_fallback_structure(chunks)

    def _generic_structure_prompt(self, chunks_summary: str, doc_type: DocumentType) -> str:
        """Build LLM prompt for generic document structure detection."""
        return f"""Analyze this document's structure and identify the hierarchical sections.

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

Return valid JSON array only."""

    def _earnings_call_structure_prompt(self, chunks_summary: str, num_chunks: int) -> str:
        """Build LLM prompt for earnings call structure detection.

        Produces 3-6 high-level sections (Opening, Prepared Remarks, Q&A, Closing)
        rather than per-speaker nodes.
        """
        return f"""Analyze this earnings call transcript and identify 3 to 6 HIGH-LEVEL sections.

The transcript has been split into {num_chunks} chunks. Here are the chunk markers and their starting content:

{chunks_summary}

Produce ONLY high-level phases of the call. Typical sections are:
- Opening / Operator Introduction
- Prepared Remarks (management presentations)
- Q&A Session
- Closing Remarks

Do NOT create a separate node for each speaker turn. Group consecutive speaker turns
into the phase they belong to.

Coverage rules:
- The first section must start at chunk_index 0.
- The last section must end at end_chunk_index {num_chunks}.
- Sections must not overlap and must not leave gaps between them.

Return a JSON array with 3 to 6 objects:
[
  {{"structure": "1", "title": "Opening", "chunk_index": 0, "end_chunk_index": 2}},
  {{"structure": "2", "title": "Prepared Remarks", "chunk_index": 2, "end_chunk_index": 5}},
  {{"structure": "3", "title": "Q&A Session", "chunk_index": 5, "end_chunk_index": 7}},
  {{"structure": "4", "title": "Closing Remarks", "chunk_index": 7, "end_chunk_index": 8}}
]

Return valid JSON array only."""
    
    def _build_chunks_summary(self, chunks: list[Chunk]) -> str:
        """Build summary of chunks for LLM, focusing on section boundaries.

        Shows all chunks with section titles (structural boundaries) with previews,
        and collapses runs of content-only chunks into counts.  This ensures the LLM
        sees the full document structure regardless of document size.
        """
        lines = []
        content_run = 0  # count of consecutive chunks without section_title

        for i, chunk in enumerate(chunks):
            if chunk.section_title:
                # Flush any accumulated content-only chunks
                if content_run > 0:
                    lines.append(f"  ... {content_run} content chunk(s) ...")
                    content_run = 0
                preview = chunk.text[:200].replace('\n', ' ').strip()
                lines.append(f"<chunk_{i}> [{chunk.section_title}]: {preview}...")
            else:
                content_run += 1

        # Flush trailing content-only chunks
        if content_run > 0:
            lines.append(f"  ... {content_run} content chunk(s) ...")

        lines.append(f"\nTotal: {len(chunks)} chunks")
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
        """Generate summaries for all nodes using batched LLM calls with fallback"""
        try:
            await self._generate_summaries_batched(nodes, chunks)
        except Exception as e:
            logger.warning(f"Batched summary generation failed: {e}, falling back to individual calls")
            await self._generate_summaries_individual(nodes, chunks)

    async def _generate_summaries_batched(
        self,
        nodes: list[TreeNode],
        chunks: list[str],
    ) -> None:
        """Generate summaries using batched LLM calls with bottom-up parent synthesis.

        Leaf nodes are summarized from raw text via batched LLM calls.
        Parent nodes are then synthesized from their children's summaries,
        avoiding redundant processing of overlapping text.
        """
        all_nodes: list[TreeNode] = []
        self._collect_all_nodes(nodes, all_nodes)

        # Separate leaf nodes (no children) from parent nodes (have children)
        leaf_nodes = []
        parent_nodes = []
        for node in all_nodes:
            text = "\n".join(chunks[node.start_index:node.end_index])
            if len(text) <= 100:
                continue
            if node.children:
                parent_nodes.append(node)
            else:
                leaf_nodes.append(node)

        # Phase 1: Summarize leaf nodes from raw text via batched LLM calls
        if leaf_nodes:
            batches = self._create_token_aware_batches(leaf_nodes, chunks)
            logger.info(
                f"Generating summaries for {len(leaf_nodes)} leaf nodes "
                f"in {len(batches)} batches"
            )

            semaphore = asyncio.Semaphore(self.config.max_concurrent_batches)

            async def process_batch(batch: list[TreeNode]) -> dict[str, str]:
                async with semaphore:
                    return await self._generate_batch_summaries(batch, chunks)

            results = await asyncio.gather(
                *[process_batch(b) for b in batches], return_exceptions=True
            )

            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Batch failed: {result}")
                    continue
                for node_id, summary in result.items():
                    node = self._find_node_by_id(all_nodes, node_id)
                    if node:
                        node.summary = summary

        # Phase 2: Synthesize parent summaries from children (bottom-up by level)
        if parent_nodes:
            # Sort deepest first so children are summarized before parents
            parent_nodes.sort(key=lambda n: n.level, reverse=True)
            logger.info(f"Synthesizing summaries for {len(parent_nodes)} parent nodes")

            for node in parent_nodes:
                child_summaries = [
                    f"- {c.title}: {c.summary}"
                    for c in node.children
                    if c.summary
                ]
                if child_summaries:
                    prompt = (
                        f"Synthesize a 2-3 sentence summary for the section "
                        f'"{node.title}" based on its subsections:\n\n'
                        + "\n".join(child_summaries)
                        + "\n\nProvide a concise summary:"
                    )
                    try:
                        summary = await self._cached_llm_complete(
                            prompt, llm=self.summary_llm
                        )
                        node.summary = summary.strip()[:500]
                    except Exception as e:
                        logger.warning(f"Parent synthesis failed for {node.title}: {e}")
                        # Fallback: concatenate child summaries
                        node.summary = " ".join(
                            c.summary for c in node.children if c.summary
                        )[:500]
                else:
                    # No child summaries available, summarize from raw text
                    text = "\n".join(chunks[node.start_index:node.end_index])
                    node.summary = await self._generate_summary(text, node.title)

    def _create_token_aware_batches(
        self,
        nodes: list[TreeNode],
        chunks: list[str],
    ) -> list[list[TreeNode]]:
        """Create batches respecting both token budget and max batch size."""
        batches: list[list[TreeNode]] = []
        current_batch: list[TreeNode] = []
        current_tokens = 0
        chars_per_token = 3.5
        max_sample = 2000  # matches _generate_batch_summaries sample size

        for node in nodes:
            text = "\n".join(chunks[node.start_index:node.end_index])
            sampled_len = min(len(text), max_sample)
            estimated_tokens = int(sampled_len / chars_per_token)

            would_exceed_tokens = (
                current_tokens + estimated_tokens > self.config.summary_token_budget
            )
            would_exceed_size = len(current_batch) >= self.config.summary_batch_size

            if current_batch and (would_exceed_tokens or would_exceed_size):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(node)
            current_tokens += estimated_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    async def _generate_batch_summaries(
        self,
        batch: list[TreeNode],
        chunks: list[str],
    ) -> dict[str, str]:
        """Generate summaries for a batch of nodes in a single LLM call"""
        if not batch:
            return {}

        # Build batch prompt with section contents
        sections_text = []
        for node in batch:
            text = "\n".join(chunks[node.start_index:node.end_index])
            text = self._sample_text(text, max_chars=2000)
            sections_text.append(f"[{node.node_id}] {node.title}:\n{text}")

        prompt = f"""Summarize each of the following document sections in 2-3 sentences each.
Focus on the key information and main points.

Sections:
{"---".join(sections_text)}

Return a JSON array with summaries for each section:
[
  {{"node_id": "0001", "summary": "2-3 sentence summary..."}},
  {{"node_id": "0002", "summary": "2-3 sentence summary..."}}
]

Provide summaries for ALL {len(batch)} sections. Return valid JSON array only."""

        try:
            results = await self._cached_llm_complete_json(prompt, llm=self.summary_llm)
            if not isinstance(results, list):
                logger.warning("Batch summary returned non-list, returning empty")
                return {}

            return {
                r["node_id"]: str(r.get("summary", ""))[:500]
                for r in results
                if isinstance(r, dict) and "node_id" in r
            }
        except Exception as e:
            logger.warning(f"Batch summary generation failed: {e}")
            return {}

    async def _generate_summaries_individual(
        self,
        nodes: list[TreeNode],
        chunks: list[str],
    ) -> None:
        """Generate summaries individually (fallback method)"""
        all_nodes: list[TreeNode] = []
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

    def _find_node_by_id(self, nodes: list[TreeNode], node_id: str) -> TreeNode | None:
        """Find a node by its ID in a flat list"""
        for node in nodes:
            if node.node_id == node_id:
                return node
        return None

    async def _generate_summary(self, text: str, title: str) -> str:
        """Generate summary for a single section"""
        text = self._sample_text(text, max_chars=6000)

        prompt = f"""Summarize this document section in 2-3 sentences.

Section: {title}

Content:
{text}

Provide a concise summary focusing on the key information:"""

        try:
            summary = await self._cached_llm_complete(prompt, llm=self.summary_llm)
            return summary.strip()[:500]  # Limit summary length
        except Exception as e:
            logger.warning(f"Summary generation failed for {title}: {e}")
            return ""

    def _sample_text(self, text: str, max_chars: int = 6000) -> str:
        """Intelligently sample text for summarization with sentence boundaries"""
        if len(text) <= max_chars:
            return text

        # Calculate space for head and tail
        half = (max_chars - 30) // 2  # Reserve space for truncation marker

        # Find sentence boundary for head
        head = text[:half + 200]
        sentence_end = max(
            head.rfind('. '),
            head.rfind('? '),
            head.rfind('! '),
        )
        if sentence_end > half // 2:
            head = head[:sentence_end + 1]
        else:
            head = text[:half]

        # Find sentence boundary for tail
        tail_start = len(text) - half - 200
        if tail_start < 0:
            tail_start = 0
        tail = text[tail_start:]
        sentence_start = tail.find('. ')
        if sentence_start > 0 and sentence_start < 400:
            tail = tail[sentence_start + 2:]
        else:
            tail = text[-half:]

        return head.strip() + "\n\n...[content truncated]...\n\n" + tail.strip()

    async def _cached_llm_complete(self, prompt: str, llm: Optional[LLMClient] = None) -> str:
        """LLM completion with optional caching"""
        client = llm or self.llm
        if self.config.use_cache and self.cache:
            model = client.config.model
            cached = await self.cache.get_llm_response(prompt, model)
            if cached is not None:
                logger.debug("Cache hit for LLM prompt")
                return cached

        response = await client.complete(prompt)

        if self.config.use_cache and self.cache:
            await self.cache.set_llm_response(prompt, client.config.model, response)

        return response

    async def _cached_llm_complete_json(self, prompt: str, llm: Optional[LLMClient] = None) -> Any:
        """JSON LLM completion with optional caching.

        Caches the parsed JSON (serialized via json.dumps) so cache hits
        skip the extraction/fixup logic entirely.
        """
        client = llm or self.llm
        cache_key_suffix = ":json"  # Distinguish from raw text cache entries

        if self.config.use_cache and self.cache:
            model = client.config.model
            cached = await self.cache.get_llm_response(
                prompt + cache_key_suffix, model
            )
            if cached is not None:
                logger.debug("Cache hit for LLM JSON prompt")
                try:
                    return json.loads(cached)
                except json.JSONDecodeError:
                    pass  # Corrupted cache entry, re-fetch

        response = await client.complete(prompt, response_format="json")
        parsed = client._extract_json(response)

        if self.config.use_cache and self.cache:
            # Store the clean parsed JSON so we skip _extract_json on cache hits
            await self.cache.set_llm_response(
                prompt + cache_key_suffix,
                client.config.model,
                json.dumps(parsed),
            )

        return parsed


# ============================================================================
# Convenience functions
# ============================================================================

async def index_document(
    text: str,
    doc_name: str = "document",
    doc_type: Optional[DocumentType] = None,
    model: str = "gpt-4o",
    summary_model: Optional[str] = None,
) -> DocumentIndex:
    """
    Convenience function to index a document.

    Args:
        text: Document text
        doc_name: Document name
        doc_type: Document type (auto-detected if not provided)
        model: LLM model for structure detection (e.g. "gpt-4o", "bedrock/anthropic.claude-sonnet-4-5-20250929")
        summary_model: Optional cheaper model for summaries (e.g. "gpt-4o-mini", "bedrock/anthropic.claude-haiku-4-5-20251001")

    Returns:
        DocumentIndex
    """
    config = IndexerConfig(
        llm_config=LLMConfig(model=model),
        summary_llm_config=LLMConfig(model=summary_model) if summary_model else None,
    )
    indexer = DocumentIndexer(config)
    return await indexer.index(text, doc_name, doc_type)
