"""
Node search functionality - finds all nodes related to a query.

This is the foundation for both:
- Agentic QA (selective, iterative retrieval)
- Provenance Extraction (exhaustive scan)
"""

from dataclasses import dataclass
from typing import Optional, Any
import asyncio
import json
import logging

from .models import DocumentIndex, TreeNode, NodeMatch
from .llm_client import LLMClient, LLMConfig
from .cache import CacheManager
from .streaming import ProgressCallback, ProgressUpdate, OperationType

logger = logging.getLogger(__name__)


@dataclass
class NodeSearchConfig:
    """Configuration for node search"""
    relevance_threshold: float = 0.5  # Minimum relevance score (0-1)
    max_results: int = 20
    include_children: bool = True  # Include children of matching nodes
    follow_cross_refs: bool = True  # Follow cross-references
    use_cache: bool = True
    batch_size: int = 10  # Nodes to score per LLM call
    max_concurrent_batches: int = 3  # Parallel batch processing limit


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
        llm_client: Optional[LLMClient] = None,
        llm_config: Optional[LLMConfig] = None,
        cache_manager: Optional[CacheManager] = None,
    ):
        self.doc_index = doc_index
        self.llm = llm_client or LLMClient(llm_config or LLMConfig())
        self.cache = cache_manager

    async def find_related_nodes(
        self,
        query: str,
        config: Optional[NodeSearchConfig] = None,
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
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return cached

        # Get all nodes to evaluate
        all_nodes = self.doc_index.get_all_nodes()

        if not all_nodes:
            return []

        # Score nodes in batches
        matches = await self._score_nodes_batched(all_nodes, query, config)

        # Filter by threshold and sort
        matches = [m for m in matches if m.relevance_score >= config.relevance_threshold]
        matches.sort(key=lambda m: m.relevance_score, reverse=True)

        # Limit results
        matches = matches[:config.max_results]

        # Follow cross-references if enabled
        if config.follow_cross_refs and matches:
            matches = await self._expand_with_cross_refs(matches, query, config)

        # Cache results
        if config.use_cache and self.cache:
            await self.cache.set_search_result(self.doc_index.doc_id, query, matches)

        return matches

    async def find_related_nodes_with_progress(
        self,
        query: str,
        config: Optional[NodeSearchConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> list[NodeMatch]:
        """Find related nodes with progress reporting"""
        config = config or NodeSearchConfig()

        all_nodes = self.doc_index.get_all_nodes()
        total_batches = (len(all_nodes) + config.batch_size - 1) // config.batch_size

        matches: list[NodeMatch] = []

        for i in range(0, len(all_nodes), config.batch_size):
            batch = all_nodes[i:i + config.batch_size]
            batch_matches = await self._score_batch(batch, query)
            matches.extend(batch_matches)

            if progress_callback:
                batch_num = i // config.batch_size + 1
                progress_callback(ProgressUpdate(
                    operation=OperationType.SEARCHING,
                    current_step=batch_num,
                    total_steps=total_batches,
                    step_name=f"Scoring batch {batch_num}/{total_batches}",
                    message=f"Found {len([m for m in matches if m.relevance_score >= config.relevance_threshold])} relevant nodes",
                ))

        # Filter and sort
        matches = [m for m in matches if m.relevance_score >= config.relevance_threshold]
        matches.sort(key=lambda m: m.relevance_score, reverse=True)
        matches = matches[:config.max_results]

        return matches

    async def _score_nodes_batched(
        self,
        nodes: list[TreeNode],
        query: str,
        config: NodeSearchConfig,
    ) -> list[NodeMatch]:
        """Score all nodes in batches"""
        batch_size = config.batch_size
        batches = [nodes[i:i + batch_size] for i in range(0, len(nodes), batch_size)]

        all_matches: list[NodeMatch] = []

        semaphore = asyncio.Semaphore(config.max_concurrent_batches)

        async def process_batch(batch: list[TreeNode]) -> list[NodeMatch]:
            async with semaphore:
                return await self._score_batch(batch, query)

        results = await asyncio.gather(*[process_batch(b) for b in batches])

        for batch_matches in results:
            all_matches.extend(batch_matches)

        return all_matches

    async def _score_batch(
        self,
        nodes: list[TreeNode],
        query: str,
    ) -> list[NodeMatch]:
        """Score a batch of nodes for relevance"""
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

        prompt = f"""Evaluate how relevant each document section is to this query/topic.

Query: {query}

Sections to evaluate:
{nodes_text}

For each section, provide a relevance score (0.0-1.0) and brief reasoning.

Score guidelines:
- 0.9-1.0: Directly addresses the query
- 0.7-0.8: Highly relevant, contains key information
- 0.5-0.6: Somewhat relevant, may contain useful context
- 0.3-0.4: Tangentially related
- 0.0-0.2: Not relevant

Return JSON array:
[
  {{"node_id": "0001", "score": 0.8, "reason": "Contains discussion of..."}},
  {{"node_id": "0002", "score": 0.2, "reason": "Not related to query"}}
]

Include ALL sections in your response."""

        try:
            results = await self._cached_llm_complete_json(prompt)

            if not isinstance(results, list):
                results = []

            # Build result map
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
                    # Node not in results, assign low score
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
            # Return all with neutral score on error
            return [
                NodeMatch(node=n, relevance_score=0.3, match_reason="Scoring error")
                for n in nodes
            ]

    async def _expand_with_cross_refs(
        self,
        matches: list[NodeMatch],
        query: str,
        config: NodeSearchConfig,
    ) -> list[NodeMatch]:
        """Expand results by following cross-references (batched)"""
        matched_ids = {m.node.node_id for m in matches}
        targets_to_score: list[TreeNode] = []

        # Collect all unscored cross-referenced nodes
        for match in matches:
            for ref in match.node.cross_references:
                if ref.resolved and ref.target_node_id and ref.target_node_id not in matched_ids:
                    target = self.doc_index.find_node(ref.target_node_id)
                    if target:
                        targets_to_score.append(target)
                        matched_ids.add(target.node_id)  # Prevent duplicates

        if not targets_to_score:
            return matches

        # Score all cross-referenced nodes in batch(es)
        additional: list[NodeMatch] = []
        batch_size = config.batch_size
        for i in range(0, len(targets_to_score), batch_size):
            batch = targets_to_score[i:i + batch_size]
            ref_matches = await self._score_batch(batch, query)
            for m in ref_matches:
                if m.relevance_score >= config.relevance_threshold:
                    additional.append(m)

        return matches + additional

    # -------------------------------------------------------------------------
    # Caching
    # -------------------------------------------------------------------------

    async def _cached_llm_complete_json(
        self,
        prompt: str,
        llm: Optional[LLMClient] = None,
    ) -> Any:
        """JSON LLM completion with caching.

        Caches the parsed JSON so cache hits skip extraction/fixup logic.
        """
        client = llm or self.llm
        cache_key_suffix = ":json"

        if self.cache:
            model = client.config.model
            cached = await self.cache.get_llm_response(
                prompt + cache_key_suffix, model
            )
            if cached is not None:
                logger.debug("Cache hit for searcher LLM JSON prompt")
                try:
                    return json.loads(cached)
                except json.JSONDecodeError:
                    pass  # Corrupted cache entry, re-fetch

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
                logger.debug("Cache hit for searcher LLM prompt")
                return cached

        response = await client.complete(prompt)

        if self.cache:
            await self.cache.set_llm_response(prompt, client.config.model, response)

        return response

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_node_with_context(
        self,
        node_id: str,
        context_chunks: int = 1,
    ) -> Optional[str]:
        """
        Get node text with surrounding context.

        Args:
            node_id: Node ID to retrieve
            context_chunks: Number of chunks of context on each side

        Returns:
            Text with context
        """
        node = self.doc_index.find_node(node_id)
        if not node:
            return None

        start = max(0, node.start_index - context_chunks)
        end = min(len(self.doc_index.chunks), node.end_index + context_chunks)

        return self.doc_index.get_chunk_text(start, end)

    def get_node_text(self, node_id: str) -> Optional[str]:
        """Get full text for a node"""
        return self.doc_index.get_node_text(node_id)


# ============================================================================
# Convenience function
# ============================================================================

async def search_nodes(
    doc_index: DocumentIndex,
    query: str,
    threshold: float = 0.5,
    max_results: int = 20,
    model: str = "gpt-4o",
) -> list[NodeMatch]:
    """
    Convenience function to search for related nodes.

    Args:
        doc_index: Document index to search
        query: Search query
        threshold: Minimum relevance score
        max_results: Maximum results to return
        model: LLM model to use

    Returns:
        List of matching nodes
    """
    searcher = NodeSearcher(
        doc_index,
        llm_config=LLMConfig(model=model),
    )
    config = NodeSearchConfig(
        relevance_threshold=threshold,
        max_results=max_results,
    )
    return await searcher.find_related_nodes(query, config)
