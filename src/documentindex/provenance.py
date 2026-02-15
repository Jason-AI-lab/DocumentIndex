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

from dataclasses import dataclass, field
from typing import Optional, AsyncIterator
import asyncio
import json
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
    parallel_workers: int = 5  # Concurrent batch processing
    batch_size: int = 10  # Nodes per LLM batch

    # Excerpt extraction optimization
    excerpt_threshold: float = 0.75  # Only extract excerpts for high-confidence matches
    excerpt_token_budget: int = 30000  # Max input tokens per excerpt batch

    # Multi-model support
    scoring_llm_config: Optional[LLMConfig] = None  # Cheaper model for scoring + summary

    # Concurrency
    max_concurrent_categories: int = 3  # Parallel category extractions


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
        scoring_llm_config: Optional[LLMConfig] = None,
    ):
        self.doc_index = doc_index
        self.llm = llm_client or LLMClient(llm_config or LLMConfig())
        self.scoring_llm = (
            LLMClient(scoring_llm_config)
            if scoring_llm_config
            else self.llm
        )
        self.cache = cache_manager
        self.searcher = NodeSearcher(
            doc_index,
            llm_client=self.scoring_llm,
            cache_manager=cache_manager,
        )

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

        # Extract excerpts for matching nodes (batched, with threshold)
        if config.extract_excerpts:
            await self._extract_excerpts_batched(matches, topic, config)

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

        # Score nodes in batches (using searcher — consolidated, cached)
        matches: list[NodeMatch] = []

        for i in range(0, total_nodes, config.batch_size):
            batch = all_nodes[i:i + config.batch_size]
            batch_matches = await self.searcher._score_batch(batch, topic)

            for match in batch_matches:
                if match.relevance_score >= config.relevance_threshold:
                    matches.append(match)

            reporter.report(
                f"Scanning batch {i // config.batch_size + 1}/{total_batches}",
                f"Found {len(matches)} relevant nodes",
            )

        # Extract excerpts (batched)
        reporter.report("Extracting excerpts", f"Processing {len(matches)} nodes")
        if config.extract_excerpts and matches:
            await self._extract_excerpts_batched(matches, topic, config)

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

                # Score batch (using searcher — consolidated, cached)
                batch_matches = await self.searcher._score_batch(batch, topic)

                # Collect relevant matches for batched excerpt extraction
                relevant = [
                    m for m in batch_matches
                    if m.relevance_score >= config.relevance_threshold
                ]

                # Extract excerpts for this batch's relevant matches
                if config.extract_excerpts and relevant:
                    await self._extract_excerpts_batched(relevant, topic, config)

                for match in relevant:
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

        max_concurrent = config.max_concurrent_categories
        semaphore = asyncio.Semaphore(max_concurrent)

        async def extract_category(name: str, topic: str) -> tuple[str, ProvenanceResult]:
            async with semaphore:
                result = await self.extract_all(topic, config)
                return name, result

        tasks = [extract_category(name, topic) for name, topic in categories.items()]
        results = await asyncio.gather(*tasks)

        return dict(results)

    # -------------------------------------------------------------------------
    # Excerpt Extraction (batched, full content, with threshold)
    # -------------------------------------------------------------------------

    async def _extract_excerpts_batched(
        self,
        matches: list[NodeMatch],
        topic: str,
        config: ProvenanceConfig,
    ) -> None:
        """Extract excerpts using token-aware batched LLM calls.

        - Only extracts for matches above excerpt_threshold
        - Sends full node content (no truncation)
        - Groups multiple nodes per LLM call respecting token budget
        """
        # Filter to high-confidence matches for excerpt extraction
        excerpt_matches = [
            m for m in matches
            if m.relevance_score >= config.excerpt_threshold
        ]

        if not excerpt_matches:
            return

        # Create token-aware batches
        batches = self._create_excerpt_batches(excerpt_matches, config)
        logger.info(
            f"Extracting excerpts for {len(excerpt_matches)} nodes "
            f"in {len(batches)} batches"
        )

        semaphore = asyncio.Semaphore(config.parallel_workers)

        async def process_batch(
            batch: list[NodeMatch],
        ) -> dict[str, list[str]]:
            async with semaphore:
                return await self._extract_excerpts_batch(batch, topic, config)

        results = await asyncio.gather(
            *[process_batch(b) for b in batches],
            return_exceptions=True,
        )

        # Apply results to matches
        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"Excerpt batch failed: {result}")
                continue
            for node_id, excerpts in result.items():
                for match in excerpt_matches:
                    if match.node.node_id == node_id:
                        match.matched_excerpts = excerpts
                        break

    def _create_excerpt_batches(
        self,
        matches: list[NodeMatch],
        config: ProvenanceConfig,
    ) -> list[list[NodeMatch]]:
        """Group matches into batches respecting token budget and max batch size."""
        batches: list[list[NodeMatch]] = []
        current_batch: list[NodeMatch] = []
        current_tokens = 0
        chars_per_token = 3.5

        for match in matches:
            text = self.doc_index.get_node_text(match.node.node_id) or ""
            estimated_tokens = int(len(text) / chars_per_token)

            would_exceed_tokens = (
                current_tokens + estimated_tokens > config.excerpt_token_budget
            )
            would_exceed_size = len(current_batch) >= config.batch_size

            if current_batch and (would_exceed_tokens or would_exceed_size):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(match)
            current_tokens += estimated_tokens

        if current_batch:
            batches.append(current_batch)

        return batches

    async def _extract_excerpts_batch(
        self,
        batch: list[NodeMatch],
        topic: str,
        config: ProvenanceConfig,
    ) -> dict[str, list[str]]:
        """Extract excerpts for a batch of nodes in a single LLM call.

        Sends full node content and returns excerpts keyed by node_id.
        """
        if not batch:
            return {}

        # Build prompt with full content for each node
        sections = []
        for match in batch:
            text = self.doc_index.get_node_text(match.node.node_id) or ""
            sections.append(
                f"=== Section [{match.node.node_id}] {match.node.title} ===\n{text}"
            )

        sections_text = "\n\n".join(sections)

        prompt = f"""Extract specific excerpts from each section that relate to the topic.

Topic: {topic}

{sections_text}

For each section, find up to {config.max_excerpts_per_node} specific passages that discuss or relate to the topic.
Quote the exact text (keep each excerpt under 200 words).

Return JSON array:
[
  {{"node_id": "{batch[0].node.node_id}", "excerpts": ["exact quote...", "another passage..."]}},
  {{"node_id": "{batch[-1].node.node_id}", "excerpts": ["exact quote..."]}}
]

Include ALL sections. Return empty excerpts array if no relevant passages found in a section."""

        try:
            result = await self._cached_llm_complete_json(prompt)

            if not isinstance(result, list):
                return {}

            excerpts_map: dict[str, list[str]] = {}
            for item in result:
                if isinstance(item, dict) and "node_id" in item:
                    node_id = item["node_id"]
                    raw_excerpts = item.get("excerpts", [])
                    if isinstance(raw_excerpts, list):
                        excerpts_map[node_id] = [
                            str(e)[:1000]
                            for e in raw_excerpts[:config.max_excerpts_per_node]
                        ]

            return excerpts_map

        except Exception as e:
            logger.warning(f"Batch excerpt extraction failed: {e}")
            # Fallback: try individual extraction
            fallback_results: dict[str, list[str]] = {}
            for match in batch:
                try:
                    individual = await self._extract_excerpts_single(
                        match, topic, config
                    )
                    if individual:
                        fallback_results[match.node.node_id] = individual
                except Exception as inner_e:
                    logger.warning(
                        f"Individual excerpt extraction failed for "
                        f"{match.node.node_id}: {inner_e}"
                    )
            return fallback_results

    async def _extract_excerpts_single(
        self,
        match: NodeMatch,
        topic: str,
        config: ProvenanceConfig,
    ) -> list[str]:
        """Extract excerpts from a single node (fallback for failed batches)."""
        text = self.doc_index.get_node_text(match.node.node_id)
        if not text:
            return []

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
            result = await self._cached_llm_complete_json(prompt)
            excerpts = result.get("excerpts", [])
            if isinstance(excerpts, list):
                return [str(e)[:1000] for e in excerpts[:config.max_excerpts_per_node]]
        except Exception as e:
            logger.warning(f"Excerpt extraction failed for {match.node.node_id}: {e}")

        return []

    # -------------------------------------------------------------------------
    # Summary Generation
    # -------------------------------------------------------------------------

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
            return await self._cached_llm_complete(prompt, llm=self.scoring_llm)
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            return f"Found {len(matches)} relevant sections discussing {topic}."

    # -------------------------------------------------------------------------
    # Caching
    # -------------------------------------------------------------------------

    async def _cached_llm_complete_json(
        self,
        prompt: str,
        llm: Optional[LLMClient] = None,
    ):
        """JSON LLM completion with caching."""
        client = llm or self.llm
        cache_key_suffix = ":json"

        if self.cache:
            model = client.config.model
            cached = await self.cache.get_llm_response(
                prompt + cache_key_suffix, model
            )
            if cached is not None:
                logger.debug("Cache hit for provenance LLM JSON prompt")
                try:
                    return json.loads(cached)
                except json.JSONDecodeError:
                    pass

        response = await client.complete(prompt, response_format="json")
        parsed = client._extract_json(response)

        if self.cache:
            await self.cache.set_llm_response(
                prompt + cache_key_suffix,
                client.config.model,
                json.dumps(parsed),
            )

        return parsed

    async def _cached_llm_complete(
        self,
        prompt: str,
        llm: Optional[LLMClient] = None,
    ) -> str:
        """LLM completion with caching."""
        client = llm or self.llm
        if self.cache:
            model = client.config.model
            cached = await self.cache.get_llm_response(prompt, model)
            if cached is not None:
                logger.debug("Cache hit for provenance LLM prompt")
                return cached

        response = await client.complete(prompt)

        if self.cache:
            await self.cache.set_llm_response(prompt, client.config.model, response)

        return response


# ============================================================================
# Convenience functions
# ============================================================================

async def extract_provenance(
    doc_index: DocumentIndex,
    topic: str,
    threshold: float = 0.6,
    model: str = "gpt-4o",
    scoring_model: Optional[str] = None,
) -> ProvenanceResult:
    """
    Convenience function to extract provenance.

    Args:
        doc_index: Document index to search
        topic: Topic to find evidence for
        threshold: Minimum relevance score
        model: LLM model for excerpt extraction
        scoring_model: Optional cheaper model for scoring + summary

    Returns:
        ProvenanceResult with all evidence
    """
    extractor = ProvenanceExtractor(
        doc_index,
        llm_config=LLMConfig(model=model),
        scoring_llm_config=LLMConfig(model=scoring_model) if scoring_model else None,
    )
    config = ProvenanceConfig(relevance_threshold=threshold)
    return await extractor.extract_all(topic, config)


async def extract_multiple_topics(
    doc_index: DocumentIndex,
    topics: dict[str, str],
    threshold: float = 0.6,
    model: str = "gpt-4o",
    scoring_model: Optional[str] = None,
) -> dict[str, ProvenanceResult]:
    """
    Extract provenance for multiple topics.

    Args:
        doc_index: Document index to search
        topics: Dict of {name: topic_description}
        threshold: Minimum relevance score
        model: LLM model for excerpt extraction
        scoring_model: Optional cheaper model for scoring + summary

    Returns:
        Dict of {name: ProvenanceResult}
    """
    extractor = ProvenanceExtractor(
        doc_index,
        llm_config=LLMConfig(model=model),
        scoring_llm_config=LLMConfig(model=scoring_model) if scoring_model else None,
    )
    config = ProvenanceConfig(relevance_threshold=threshold)
    return await extractor.extract_by_category(topics, config)
