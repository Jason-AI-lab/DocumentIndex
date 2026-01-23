"""
Provenance Extraction - exhaustive scan to find ALL evidence related to a topic.

Unlike agentic QA which stops when sufficient information is found,
provenance extraction scans the entire document to find every piece
of relevant evidence.

Use cases:
- Compliance audits
- Risk factor analysis
- Theme extraction across entire documents
- Evidence gathering for reports
"""

from dataclasses import dataclass
from typing import Optional, AsyncIterator
import asyncio
import logging

from .models import DocumentIndex, TreeNode, NodeMatch, ProvenanceResult
from .llm_client import LLMClient, LLMConfig
from .searcher import NodeSearcher, NodeSearchConfig
from .cache import CacheManager
from .streaming import (
    StreamingProvenanceResult,
    ProgressCallback, ProgressUpdate, OperationType, ProgressReporter,
)

logger = logging.getLogger(__name__)


@dataclass
class ProvenanceConfig:
    """Configuration for provenance extraction"""
    relevance_threshold: float = 0.6  # Higher threshold for quality
    extract_excerpts: bool = True
    max_excerpts_per_node: int = 5
    generate_summary: bool = True
    parallel_workers: int = 5  # Concurrent node processing
    batch_size: int = 10  # Nodes per LLM batch


class ProvenanceExtractor:
    """
    Exhaustive provenance extraction from documents.
    
    Scans ALL nodes in the document to find every piece of
    evidence related to a topic.
    """
    
    def __init__(
        self,
        doc_index: DocumentIndex,
        llm_client: Optional[LLMClient] = None,
        llm_config: Optional[LLMConfig] = None,
        cache_manager: Optional[CacheManager] = None,
    ):
        self.doc_index = doc_index
        self.llm = llm_client or LLMClient(llm_config or LLMConfig())
        self.cache = cache_manager
        self.searcher = NodeSearcher(doc_index, llm_client=self.llm, cache_manager=cache_manager)
    
    async def extract_all(
        self,
        topic: str,
        config: Optional[ProvenanceConfig] = None,
    ) -> ProvenanceResult:
        """
        Extract all evidence related to a topic.
        
        Scans every node in the document and returns all
        nodes that contain relevant information.
        
        Args:
            topic: The topic/theme to find evidence for
            config: Extraction configuration
        
        Returns:
            ProvenanceResult with all matching evidence
        """
        config = config or ProvenanceConfig()
        
        all_nodes = self.doc_index.get_all_nodes()
        total_nodes = len(all_nodes)
        
        if total_nodes == 0:
            return ProvenanceResult(
                topic=topic,
                evidence=[],
                total_nodes_scanned=0,
                scan_coverage=0.0,
            )
        
        # Score all nodes using searcher
        search_config = NodeSearchConfig(
            relevance_threshold=config.relevance_threshold,
            max_results=total_nodes,  # Get all matching nodes
            batch_size=config.batch_size,
        )
        
        matches = await self.searcher.find_related_nodes(topic, search_config)
        
        # Extract excerpts for matching nodes
        if config.extract_excerpts:
            await self._extract_excerpts_parallel(matches, topic, config)
        
        # Generate summary
        summary = None
        if config.generate_summary and matches:
            summary = await self._generate_summary(topic, matches)
        
        return ProvenanceResult(
            topic=topic,
            evidence=matches,
            total_nodes_scanned=total_nodes,
            scan_coverage=1.0,  # Always scan everything
            summary=summary,
        )
    
    async def extract_with_progress(
        self,
        topic: str,
        config: Optional[ProvenanceConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> ProvenanceResult:
        """
        Extract provenance with progress reporting.
        
        Reports progress as nodes are scanned.
        """
        config = config or ProvenanceConfig()
        
        all_nodes = self.doc_index.get_all_nodes()
        total_nodes = len(all_nodes)
        
        if total_nodes == 0:
            return ProvenanceResult(
                topic=topic,
                evidence=[],
                total_nodes_scanned=0,
                scan_coverage=0.0,
            )
        
        # Calculate total steps: batches + excerpt extraction + summary
        total_batches = (total_nodes + config.batch_size - 1) // config.batch_size
        total_steps = total_batches + 2  # +1 for excerpts, +1 for summary
        
        reporter = ProgressReporter(
            operation=OperationType.PROVENANCE,
            total_steps=total_steps,
            callback=progress_callback,
        )
        
        # Score nodes in batches
        matches: list[NodeMatch] = []
        
        for i in range(0, total_nodes, config.batch_size):
            batch = all_nodes[i:i + config.batch_size]
            batch_matches = await self._score_batch(batch, topic)
            
            for match in batch_matches:
                if match.relevance_score >= config.relevance_threshold:
                    matches.append(match)
            
            reporter.report(
                f"Scanning batch {i // config.batch_size + 1}/{total_batches}",
                f"Found {len(matches)} relevant nodes",
            )
        
        # Extract excerpts
        reporter.report("Extracting excerpts", f"Processing {len(matches)} nodes")
        if config.extract_excerpts and matches:
            await self._extract_excerpts_parallel(matches, topic, config)
        
        # Generate summary
        summary = None
        if config.generate_summary and matches:
            reporter.report("Generating summary", "Synthesizing findings")
            summary = await self._generate_summary(topic, matches)
        else:
            reporter.report("Finalizing", "Complete")
        
        return ProvenanceResult(
            topic=topic,
            evidence=matches,
            total_nodes_scanned=total_nodes,
            scan_coverage=1.0,
            summary=summary,
        )
    
    async def extract_stream(
        self,
        topic: str,
        config: Optional[ProvenanceConfig] = None,
    ) -> StreamingProvenanceResult:
        """
        Extract provenance with streaming results.
        
        Yields NodeMatch objects as they are found, enabling
        early access to results while scanning continues.
        """
        config = config or ProvenanceConfig()
        
        all_nodes = self.doc_index.get_all_nodes()
        total_nodes = len(all_nodes)
        
        async def evidence_generator() -> AsyncIterator[NodeMatch]:
            """Generate matches as they're found"""
            for i in range(0, total_nodes, config.batch_size):
                batch = all_nodes[i:i + config.batch_size]
                
                # Score batch
                batch_matches = await self._score_batch(batch, topic)
                
                # Yield relevant matches immediately
                for match in batch_matches:
                    if match.relevance_score >= config.relevance_threshold:
                        # Extract excerpts for this match
                        if config.extract_excerpts:
                            await self._extract_excerpts_for_node(match, topic, config)
                        
                        yield match
        
        return StreamingProvenanceResult(
            topic=topic,
            evidence_stream=evidence_generator(),
            total_nodes_scanned=total_nodes,
        )
    
    async def extract_by_category(
        self,
        categories: dict[str, str],
        config: Optional[ProvenanceConfig] = None,
    ) -> dict[str, ProvenanceResult]:
        """
        Extract provenance for multiple topics/categories.
        
        Args:
            categories: Dict of {category_name: topic_description}
            config: Extraction configuration
        
        Returns:
            Dict of {category_name: ProvenanceResult}
        
        Example:
            categories = {
                "climate_risks": "climate change risks and environmental impact",
                "regulatory": "regulatory compliance and legal requirements",
                "financial_performance": "revenue, earnings, and financial metrics",
            }
            results = await extractor.extract_by_category(categories)
        """
        config = config or ProvenanceConfig()
        
        # Process categories concurrently
        semaphore = asyncio.Semaphore(3)  # Limit concurrent extractions
        
        async def extract_category(name: str, topic: str) -> tuple[str, ProvenanceResult]:
            async with semaphore:
                result = await self.extract_all(topic, config)
                return name, result
        
        tasks = [extract_category(name, topic) for name, topic in categories.items()]
        results = await asyncio.gather(*tasks)
        
        return dict(results)
    
    async def _score_batch(
        self,
        nodes: list[TreeNode],
        topic: str,
    ) -> list[NodeMatch]:
        """Score a batch of nodes for relevance to topic"""
        if not nodes:
            return []
        
        # Build node descriptions
        node_descriptions = []
        for node in nodes:
            desc = f"[{node.node_id}] {node.title}"
            if node.summary:
                desc += f" - {node.summary[:150]}"
            node_descriptions.append(desc)
        
        nodes_text = "\n".join(node_descriptions)
        
        prompt = f"""Evaluate how relevant each document section is to this topic.

Topic: {topic}

Sections to evaluate:
{nodes_text}

For each section, provide a relevance score (0.0-1.0) and brief reasoning.

Score guidelines:
- 0.9-1.0: Directly discusses this topic
- 0.7-0.8: Contains significant relevant content
- 0.5-0.6: Has some relevant mentions
- 0.3-0.4: Tangentially related
- 0.0-0.2: Not relevant

Return JSON array with ALL sections:
[
  {{"node_id": "0001", "score": 0.9, "reason": "Directly discusses..."}},
  {{"node_id": "0002", "score": 0.1, "reason": "Not related"}}
]"""

        try:
            results = await self.llm.complete_json(prompt)
            
            if not isinstance(results, list):
                results = []
            
            result_map = {}
            for r in results:
                if isinstance(r, dict) and "node_id" in r:
                    result_map[r["node_id"]] = r
            
            matches = []
            for node in nodes:
                if node.node_id in result_map:
                    r = result_map[node.node_id]
                    score = float(r.get("score", 0))
                    reason = r.get("reason", "")
                else:
                    score = 0.1
                    reason = "Not evaluated"
                
                matches.append(NodeMatch(
                    node=node,
                    relevance_score=score,
                    match_reason=reason,
                ))
            
            return matches
            
        except Exception as e:
            logger.warning(f"Batch scoring failed: {e}")
            return [
                NodeMatch(node=n, relevance_score=0.3, match_reason="Scoring error")
                for n in nodes
            ]
    
    async def _extract_excerpts_parallel(
        self,
        matches: list[NodeMatch],
        topic: str,
        config: ProvenanceConfig,
    ) -> None:
        """Extract excerpts for all matches in parallel"""
        semaphore = asyncio.Semaphore(config.parallel_workers)
        
        async def extract_one(match: NodeMatch):
            async with semaphore:
                await self._extract_excerpts_for_node(match, topic, config)
        
        await asyncio.gather(*[extract_one(m) for m in matches])
    
    async def _extract_excerpts_for_node(
        self,
        match: NodeMatch,
        topic: str,
        config: ProvenanceConfig,
    ) -> None:
        """Extract specific excerpts from a node related to the topic"""
        text = self.doc_index.get_node_text(match.node.node_id)
        if not text:
            return
        
        # Truncate if too long
        if len(text) > 8000:
            text = text[:4000] + "\n...[truncated]...\n" + text[-4000:]
        
        prompt = f"""Extract specific excerpts from this section that relate to the topic.

Topic: {topic}

Section [{match.node.node_id}] {match.node.title}:
{text}

Find up to {config.max_excerpts_per_node} specific passages that discuss or relate to the topic.
Quote the exact text (keep each excerpt under 200 words).

Return JSON:
{{
  "excerpts": [
    "exact quote from the text...",
    "another relevant passage..."
  ]
}}

Return empty array if no specific relevant passages found."""

        try:
            result = await self.llm.complete_json(prompt)
            excerpts = result.get("excerpts", [])
            if isinstance(excerpts, list):
                match.matched_excerpts = [str(e)[:1000] for e in excerpts[:config.max_excerpts_per_node]]
        except Exception as e:
            logger.warning(f"Excerpt extraction failed for {match.node.node_id}: {e}")
    
    async def _generate_summary(
        self,
        topic: str,
        matches: list[NodeMatch],
    ) -> str:
        """Generate overall summary of evidence found"""
        if not matches:
            return "No evidence found for this topic."
        
        # Build evidence summary
        evidence_text = ""
        for match in matches[:15]:  # Limit to prevent context overflow
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

        try:
            return await self.llm.complete(prompt)
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return f"Found {len(matches)} relevant sections discussing {topic}."


# ============================================================================
# Convenience functions
# ============================================================================

async def extract_provenance(
    doc_index: DocumentIndex,
    topic: str,
    threshold: float = 0.6,
    model: str = "gpt-4o",
) -> ProvenanceResult:
    """
    Convenience function to extract provenance.
    
    Args:
        doc_index: Document index to search
        topic: Topic to find evidence for
        threshold: Minimum relevance score
        model: LLM model to use
    
    Returns:
        ProvenanceResult with all evidence
    """
    extractor = ProvenanceExtractor(
        doc_index,
        llm_config=LLMConfig(model=model),
    )
    config = ProvenanceConfig(relevance_threshold=threshold)
    return await extractor.extract_all(topic, config)


async def extract_multiple_topics(
    doc_index: DocumentIndex,
    topics: dict[str, str],
    threshold: float = 0.6,
    model: str = "gpt-4o",
) -> dict[str, ProvenanceResult]:
    """
    Extract provenance for multiple topics.
    
    Args:
        doc_index: Document index to search
        topics: Dict of {name: topic_description}
        threshold: Minimum relevance score
        model: LLM model to use
    
    Returns:
        Dict of {name: ProvenanceResult}
    """
    extractor = ProvenanceExtractor(
        doc_index,
        llm_config=LLMConfig(model=model),
    )
    config = ProvenanceConfig(relevance_threshold=threshold)
    return await extractor.extract_by_category(topics, config)
