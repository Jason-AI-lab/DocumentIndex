"""Tests for provenance extraction and searcher optimizations."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json

from documentindex.provenance import (
    ProvenanceExtractor,
    ProvenanceConfig,
    extract_provenance,
    extract_multiple_topics,
)
from documentindex.searcher import NodeSearcher, NodeSearchConfig
from documentindex.llm_client import LLMClient, LLMConfig
from documentindex.models import (
    DocumentIndex,
    DocumentType,
    TreeNode,
    TextSpan,
    NodeMatch,
    CrossReference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(
    node_id: str,
    title: str,
    summary: str = "",
    start_chunk: int = 0,
    end_chunk: int = 1,
    children: list[TreeNode] | None = None,
    cross_refs: list | None = None,
) -> TreeNode:
    node = TreeNode(
        node_id=node_id,
        title=title,
        level=0,
        text_span=TextSpan(0, 100, start_chunk, end_chunk),
        summary=summary,
    )
    if children:
        node.children = children
    if cross_refs:
        node.cross_references = cross_refs
    return node


def _make_doc_index(
    nodes: list[TreeNode],
    chunks: list[str] | None = None,
) -> DocumentIndex:
    if chunks is None:
        chunks = ["chunk text " * 20] * 5
    original_text = "\n".join(chunks)

    # Compute real char offsets and fix node TextSpans to match
    offsets = []
    pos = 0
    for chunk in chunks:
        offsets.append((pos, pos + len(chunk)))
        pos += len(chunk) + 1  # +1 for \n

    # Fix node TextSpans to use real char offsets
    for node in nodes:
        sc = node.text_span.start_chunk
        ec = min(node.text_span.end_chunk, len(chunks))
        if sc < len(offsets) and ec > 0:
            node.text_span = TextSpan(
                start_char=offsets[sc][0],
                end_char=offsets[ec - 1][1],
                start_chunk=sc,
                end_chunk=ec,
            )

    doc = DocumentIndex(
        doc_id="test-doc",
        doc_name="test",
        doc_type=DocumentType.SEC_10K,
        original_text=original_text,
        chunks=chunks,
        chunk_char_offsets=offsets,
        structure=nodes,
    )
    return doc


def _make_mock_llm(responses: list[str] | None = None) -> LLMClient:
    """Create a mock LLM client that returns canned responses."""
    llm = MagicMock(spec=LLMClient)
    llm.config = LLMConfig(model="test-model")

    if responses is None:
        responses = ['[]']

    call_count = {"n": 0}

    async def _complete(prompt, system_prompt=None, response_format="text"):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        return responses[idx]

    async def _complete_json(prompt, system_prompt=None):
        resp = await _complete(prompt, system_prompt, "json")
        return json.loads(resp)

    llm.complete = AsyncMock(side_effect=_complete)
    llm.complete_json = AsyncMock(side_effect=_complete_json)
    llm._extract_json = LLMClient._extract_json
    return llm


# ---------------------------------------------------------------------------
# NodeSearcher: LLM Caching
# ---------------------------------------------------------------------------

class TestSearcherCaching:
    @pytest.mark.asyncio
    async def test_score_batch_uses_cache(self):
        """Verify _score_batch routes through _cached_llm_complete_json."""
        mock_cache = AsyncMock()
        mock_cache.get_llm_response = AsyncMock(return_value=None)
        mock_cache.set_llm_response = AsyncMock()

        scoring_response = json.dumps([
            {"node_id": "0000", "score": 0.9, "reason": "Relevant"},
        ])
        mock_llm = _make_mock_llm([scoring_response])

        nodes = [_make_node("0000", "Section 1", "Some summary")]
        doc = _make_doc_index(nodes)
        searcher = NodeSearcher(doc, llm_client=mock_llm, cache_manager=mock_cache)

        matches = await searcher._score_batch(nodes, "test topic")

        assert len(matches) == 1
        assert matches[0].relevance_score == 0.9
        # Cache should have been set
        mock_cache.set_llm_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_score_batch_cache_hit(self):
        """On cache hit, should return parsed JSON without calling LLM."""
        cached_response = json.dumps([
            {"node_id": "0000", "score": 0.8, "reason": "Cached"},
        ])
        mock_cache = AsyncMock()
        mock_cache.get_llm_response = AsyncMock(return_value=cached_response)

        mock_llm = _make_mock_llm()

        nodes = [_make_node("0000", "Section 1")]
        doc = _make_doc_index(nodes)
        searcher = NodeSearcher(doc, llm_client=mock_llm, cache_manager=mock_cache)

        matches = await searcher._score_batch(nodes, "test topic")

        assert len(matches) == 1
        assert matches[0].relevance_score == 0.8
        assert matches[0].match_reason == "Cached"
        # LLM should NOT have been called
        mock_llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# NodeSearcher: Configurable Concurrency
# ---------------------------------------------------------------------------

class TestSearcherConcurrency:
    @pytest.mark.asyncio
    async def test_configurable_max_concurrent_batches(self):
        """Verify max_concurrent_batches is wired through config."""
        scoring_response = json.dumps([
            {"node_id": "0000", "score": 0.5, "reason": "OK"},
        ])
        mock_llm = _make_mock_llm([scoring_response])

        nodes = [_make_node(f"{i:04d}", f"Section {i}") for i in range(30)]
        doc = _make_doc_index(nodes, ["chunk " * 20] * 30)
        searcher = NodeSearcher(doc, llm_client=mock_llm)

        config = NodeSearchConfig(
            batch_size=10,
            max_concurrent_batches=2,
        )
        matches = await searcher._score_nodes_batched(nodes, "test", config)
        # Should complete without error, processing 30 nodes in 3 batches with max 2 concurrent
        assert len(matches) == 30


# ---------------------------------------------------------------------------
# NodeSearcher: Batched Cross-Reference Scoring
# ---------------------------------------------------------------------------

class TestBatchedCrossRefs:
    @pytest.mark.asyncio
    async def test_cross_refs_scored_in_batch(self):
        """Multiple cross-referenced nodes should be scored in a single batch call."""
        ref1 = CrossReference(
            source_node_id="0000",
            target_description="Note 1",
            resolved=True,
            target_node_id="ref1",
            reference_text="see Note 1",
        )
        ref2 = CrossReference(
            source_node_id="0000",
            target_description="Note 2",
            resolved=True,
            target_node_id="ref2",
            reference_text="see Note 2",
        )

        main_node = _make_node("0000", "Main Section", cross_refs=[ref1, ref2])
        ref_node1 = _make_node("ref1", "Note 1", "Note content 1")
        ref_node2 = _make_node("ref2", "Note 2", "Note content 2")

        doc = _make_doc_index([main_node, ref_node1, ref_node2])

        # First call: scoring main batch, second call: scoring cross-ref batch
        main_score = json.dumps([
            {"node_id": "0000", "score": 0.9, "reason": "Main"},
            {"node_id": "ref1", "score": 0.1, "reason": "Not matched"},
            {"node_id": "ref2", "score": 0.1, "reason": "Not matched"},
        ])
        ref_score = json.dumps([
            {"node_id": "ref1", "score": 0.8, "reason": "Referenced"},
            {"node_id": "ref2", "score": 0.7, "reason": "Referenced"},
        ])
        mock_llm = _make_mock_llm([main_score, ref_score])

        searcher = NodeSearcher(doc, llm_client=mock_llm)
        config = NodeSearchConfig(
            relevance_threshold=0.5,
            follow_cross_refs=True,
            use_cache=False,
        )

        matches = await searcher.find_related_nodes("test query", config)

        # Should have main + 2 cross-ref nodes
        assert len(matches) == 3
        node_ids = {m.node.node_id for m in matches}
        assert "0000" in node_ids
        assert "ref1" in node_ids
        assert "ref2" in node_ids

        # LLM should have been called twice (1 main batch + 1 cross-ref batch)
        # not 3 times (1 main + 2 individual cross-refs)
        assert mock_llm.complete.call_count == 2


# ---------------------------------------------------------------------------
# ProvenanceExtractor: Multi-Model Support
# ---------------------------------------------------------------------------

class TestProvenanceMultiModel:
    def test_scoring_llm_defaults_to_primary(self):
        doc = _make_doc_index([_make_node("0000", "Section")])
        extractor = ProvenanceExtractor(
            doc,
            llm_client=_make_mock_llm(),
        )
        assert extractor.scoring_llm is extractor.llm

    def test_scoring_llm_separate_when_configured(self):
        doc = _make_doc_index([_make_node("0000", "Section")])
        extractor = ProvenanceExtractor(
            doc,
            llm_config=LLMConfig(model="gpt-4o"),
            scoring_llm_config=LLMConfig(model="gpt-4o-mini"),
        )
        assert extractor.scoring_llm is not extractor.llm
        assert extractor.scoring_llm.config.model == "gpt-4o-mini"
        assert extractor.llm.config.model == "gpt-4o"

    def test_searcher_uses_scoring_llm(self):
        """Verify the searcher is initialized with the scoring LLM."""
        doc = _make_doc_index([_make_node("0000", "Section")])
        extractor = ProvenanceExtractor(
            doc,
            llm_config=LLMConfig(model="gpt-4o"),
            scoring_llm_config=LLMConfig(model="gpt-4o-mini"),
        )
        assert extractor.searcher.llm.config.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_summary_uses_scoring_llm(self):
        """Verify _generate_summary routes through scoring_llm, not primary."""
        primary = _make_mock_llm(["primary summary"])
        scoring = _make_mock_llm(["scoring summary"])

        doc = _make_doc_index([_make_node("0000", "Section")])
        extractor = ProvenanceExtractor(doc)
        extractor.llm = primary
        extractor.scoring_llm = scoring

        match = NodeMatch(
            node=_make_node("0000", "Section"),
            relevance_score=0.9,
            match_reason="test",
            matched_excerpts=["some excerpt"],
        )
        result = await extractor._generate_summary("test topic", [match])

        assert result == "scoring summary"
        scoring.complete.assert_called_once()
        primary.complete.assert_not_called()


# ---------------------------------------------------------------------------
# ProvenanceExtractor: Excerpt Threshold
# ---------------------------------------------------------------------------

class TestExcerptThreshold:
    @pytest.mark.asyncio
    async def test_low_scoring_matches_skip_excerpt_extraction(self):
        """Matches below excerpt_threshold should not have excerpts extracted."""
        excerpt_response = json.dumps([
            {"node_id": "0001", "excerpts": ["high score excerpt"]},
        ])
        mock_llm = _make_mock_llm([excerpt_response])

        high_match = NodeMatch(
            node=_make_node("0001", "High Score"),
            relevance_score=0.9,
            match_reason="very relevant",
        )
        low_match = NodeMatch(
            node=_make_node("0002", "Low Score"),
            relevance_score=0.65,
            match_reason="somewhat relevant",
        )

        doc = _make_doc_index(
            [_make_node("0001", "High"), _make_node("0002", "Low")]
        )
        extractor = ProvenanceExtractor(doc, llm_client=mock_llm)

        config = ProvenanceConfig(
            excerpt_threshold=0.75,
            extract_excerpts=True,
        )
        await extractor._extract_excerpts_batched(
            [high_match, low_match], "test topic", config
        )

        # High-scoring match should have excerpts
        assert len(high_match.matched_excerpts) > 0
        # Low-scoring match should NOT have excerpts
        assert len(low_match.matched_excerpts) == 0


# ---------------------------------------------------------------------------
# ProvenanceExtractor: Token-Aware Batched Excerpts
# ---------------------------------------------------------------------------

class TestBatchedExcerptExtraction:
    def test_creates_batches_respecting_token_budget(self):
        """Large nodes should be split into separate batches."""
        big_text = "x" * 70000  # ~20K tokens
        small_text = "y" * 7000  # ~2K tokens
        chunks = [big_text, small_text, small_text, small_text, small_text]

        nodes = [_make_node(f"{i:04d}", f"Section {i}", start_chunk=i, end_chunk=i+1) for i in range(5)]
        doc = _make_doc_index(nodes, chunks)
        extractor = ProvenanceExtractor(doc, llm_client=_make_mock_llm())

        matches = [
            NodeMatch(node=nodes[i], relevance_score=0.9, match_reason="test")
            for i in range(5)
        ]

        config = ProvenanceConfig(
            excerpt_token_budget=10000,  # ~35K chars at 3.5 chars/token
            batch_size=10,
            excerpt_threshold=0.0,  # No threshold for this test
        )
        batches = extractor._create_excerpt_batches(matches, config)

        # Big node (20K tokens) exceeds 10K budget alone, should be in its own batch
        # The 4 small nodes (~2K each) should fit in 1-2 batches
        assert len(batches) >= 2
        total = sum(len(b) for b in batches)
        assert total == 5

    def test_respects_max_batch_size(self):
        """Batches should not exceed batch_size even if token budget allows."""
        chunks = ["short text " * 10] * 10

        nodes = [_make_node(f"{i:04d}", f"Section {i}", start_chunk=i, end_chunk=i+1) for i in range(10)]
        doc = _make_doc_index(nodes, chunks)
        extractor = ProvenanceExtractor(doc, llm_client=_make_mock_llm())

        matches = [
            NodeMatch(node=nodes[i], relevance_score=0.9, match_reason="test")
            for i in range(10)
        ]

        config = ProvenanceConfig(
            excerpt_token_budget=100000,  # Very large budget
            batch_size=3,  # Small batch size
            excerpt_threshold=0.0,
        )
        batches = extractor._create_excerpt_batches(matches, config)

        # 10 nodes with max batch_size=3 should create 4 batches
        assert len(batches) == 4
        assert all(len(b) <= 3 for b in batches)

    @pytest.mark.asyncio
    async def test_batch_excerpt_extraction_full_content(self):
        """Verify full node content is sent without truncation."""
        long_text = "A" * 15000  # Well over the old 8000 char truncation limit
        chunks = [long_text]

        node = _make_node("0000", "Long Section", start_chunk=0, end_chunk=1)
        doc = _make_doc_index([node], chunks)

        excerpt_response = json.dumps([
            {"node_id": "0000", "excerpts": ["found in middle"]},
        ])
        mock_llm = _make_mock_llm([excerpt_response])
        extractor = ProvenanceExtractor(doc, llm_client=mock_llm)

        match = NodeMatch(node=node, relevance_score=0.9, match_reason="test")
        config = ProvenanceConfig(excerpt_threshold=0.0)

        result = await extractor._extract_excerpts_batch([match], "test", config)

        assert "0000" in result
        assert result["0000"] == ["found in middle"]

        # Verify the prompt sent to LLM contains the full text (no truncation)
        call_args = mock_llm.complete.call_args
        prompt_sent = call_args[0][0] if call_args[0] else call_args[1].get("prompt", "")
        assert "...[truncated]..." not in prompt_sent
        # Full 15000-char text should be present
        assert "A" * 1000 in prompt_sent


# ---------------------------------------------------------------------------
# ProvenanceExtractor: Caching
# ---------------------------------------------------------------------------

class TestProvenanceCaching:
    @pytest.mark.asyncio
    async def test_excerpt_extraction_uses_cache(self):
        """Excerpt extraction should use caching."""
        mock_cache = AsyncMock()
        mock_cache.get_llm_response = AsyncMock(return_value=None)
        mock_cache.set_llm_response = AsyncMock()
        # For searcher cache methods
        mock_cache.get_search_result = AsyncMock(return_value=None)
        mock_cache.set_search_result = AsyncMock()

        excerpt_response = json.dumps([
            {"node_id": "0000", "excerpts": ["cached excerpt"]},
        ])
        mock_llm = _make_mock_llm([excerpt_response])

        node = _make_node("0000", "Section")
        doc = _make_doc_index([node])
        extractor = ProvenanceExtractor(
            doc, llm_client=mock_llm, cache_manager=mock_cache
        )

        match = NodeMatch(node=node, relevance_score=0.9, match_reason="test")
        config = ProvenanceConfig(excerpt_threshold=0.0)

        await extractor._extract_excerpts_batched([match], "topic", config)

        # Cache should have been set
        assert mock_cache.set_llm_response.call_count >= 1


# ---------------------------------------------------------------------------
# ProvenanceExtractor: Duplicated _score_batch Removed
# ---------------------------------------------------------------------------

class TestScoreBatchConsolidation:
    def test_no_score_batch_on_extractor(self):
        """ProvenanceExtractor should not have its own _score_batch method."""
        doc = _make_doc_index([_make_node("0000", "Section")])
        extractor = ProvenanceExtractor(doc, llm_client=_make_mock_llm())
        assert not hasattr(extractor, '_score_batch')

    @pytest.mark.asyncio
    async def test_extract_with_progress_uses_searcher(self):
        """extract_with_progress should route scoring through self.searcher."""
        scoring_response = json.dumps([
            {"node_id": "0000", "score": 0.9, "reason": "Relevant"},
        ])
        mock_llm = _make_mock_llm([scoring_response])

        node = _make_node("0000", "Section", "Some summary")
        doc = _make_doc_index([node])
        extractor = ProvenanceExtractor(doc, llm_client=mock_llm)

        config = ProvenanceConfig(
            extract_excerpts=False,
            generate_summary=False,
        )
        result = await extractor.extract_with_progress("test topic", config)

        assert result.total_nodes_scanned == 1
        assert len(result.evidence) == 1


# ---------------------------------------------------------------------------
# ProvenanceExtractor: Configurable Concurrency
# ---------------------------------------------------------------------------

class TestProvenanceConcurrency:
    @pytest.mark.asyncio
    async def test_configurable_category_concurrency(self):
        """extract_by_category should respect max_concurrent_categories."""
        scoring_response = json.dumps([
            {"node_id": "0000", "score": 0.9, "reason": "Relevant"},
        ])
        mock_llm = _make_mock_llm([scoring_response])

        node = _make_node("0000", "Section")
        doc = _make_doc_index([node])
        extractor = ProvenanceExtractor(doc, llm_client=mock_llm)

        config = ProvenanceConfig(
            max_concurrent_categories=1,
            extract_excerpts=False,
            generate_summary=False,
        )

        categories = {
            "topic_a": "topic A description",
            "topic_b": "topic B description",
        }
        results = await extractor.extract_by_category(categories, config)

        assert len(results) == 2
        assert "topic_a" in results
        assert "topic_b" in results


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

class TestConvenienceFunctions:
    def test_extract_provenance_accepts_scoring_model(self):
        import inspect
        sig = inspect.signature(extract_provenance)
        assert "scoring_model" in sig.parameters
        assert sig.parameters["scoring_model"].default is None

    def test_extract_multiple_topics_accepts_scoring_model(self):
        import inspect
        sig = inspect.signature(extract_multiple_topics)
        assert "scoring_model" in sig.parameters
        assert sig.parameters["scoring_model"].default is None
