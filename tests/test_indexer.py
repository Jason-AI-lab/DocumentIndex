"""Tests for document indexer optimizations."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import json

from documentindex.indexer import DocumentIndexer, IndexerConfig, index_document
from documentindex.llm_client import LLMClient, LLMConfig
from documentindex.chunker import Chunk
from documentindex.models import DocumentType, TreeNode, TextSpan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    text: str,
    index: int,
    start_char: int = 0,
    section_title: str | None = None,
    section_level: int = 0,
) -> Chunk:
    end_char = start_char + len(text)
    return Chunk(
        text=text,
        start_char=start_char,
        end_char=end_char,
        chunk_index=index,
        section_title=section_title,
        section_level=section_level,
    )


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
# Step 1: Multi-model support
# ---------------------------------------------------------------------------

class TestMultiModelSupport:
    def test_summary_llm_defaults_to_primary(self):
        indexer = DocumentIndexer(
            config=IndexerConfig(),
            llm_client=_make_mock_llm(),
        )
        assert indexer.summary_llm is indexer.llm

    def test_summary_llm_separate_when_configured(self):
        config = IndexerConfig(
            llm_config=LLMConfig(model="gpt-4o"),
            summary_llm_config=LLMConfig(model="gpt-4o-mini"),
        )
        indexer = DocumentIndexer(config=config)
        assert indexer.summary_llm is not indexer.llm
        assert indexer.summary_llm.config.model == "gpt-4o-mini"
        assert indexer.llm.config.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_summary_uses_summary_llm(self):
        """Verify _generate_summary routes through summary_llm, not primary llm."""
        primary = _make_mock_llm(["primary response"])
        summary = _make_mock_llm(["summary response"])

        indexer = DocumentIndexer(config=IndexerConfig(use_cache=False))
        indexer.llm = primary
        indexer.summary_llm = summary

        result = await indexer._generate_summary("Some long content " * 20, "Test Section")
        assert result == "summary response"
        summary.complete.assert_called_once()
        primary.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Step 2: Build structure from chunks (skip LLM)
# ---------------------------------------------------------------------------

class TestBuildStructureFromChunks:
    def test_builds_hierarchy_from_section_metadata(self):
        indexer = DocumentIndexer(
            config=IndexerConfig(use_cache=False),
            llm_client=_make_mock_llm(),
        )
        chunks = [
            _make_chunk("Part I content", 0, 0, "PART I", 1),
            _make_chunk("Item 1 business", 1, 100, "ITEM 1. BUSINESS", 2),
            _make_chunk("More business", 2, 200, None, 0),
            _make_chunk("Item 1A risks", 3, 300, "ITEM 1A. RISK FACTORS", 2),
            _make_chunk("Part II content", 4, 400, "PART II", 1),
            _make_chunk("Item 7 MDA", 5, 500, "ITEM 7. MD&A", 2),
        ]

        result = indexer._build_structure_from_chunks(chunks)

        # Should have 2 root nodes (PART I, PART II)
        assert len(result) == 2
        assert result[0].title == "PART I"
        assert result[1].title == "PART II"

        # PART I should have children
        assert len(result[0].children) == 2
        assert result[0].children[0].title == "ITEM 1. BUSINESS"
        assert result[0].children[1].title == "ITEM 1A. RISK FACTORS"

        # PART II should have children
        assert len(result[1].children) == 1
        assert result[1].children[0].title == "ITEM 7. MD&A"

    def test_falls_back_when_no_sections(self):
        indexer = DocumentIndexer(
            config=IndexerConfig(use_cache=False),
            llm_client=_make_mock_llm(),
        )
        chunks = [
            _make_chunk("Plain text chunk 1", 0, 0),
            _make_chunk("Plain text chunk 2", 1, 100),
        ]

        result = indexer._build_structure_from_chunks(chunks)
        # Should fall back to _build_fallback_structure
        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_skips_llm_for_well_sectioned_docs(self):
        """When enough section metadata exists, _build_structure should not call LLM."""
        mock_llm = _make_mock_llm()
        indexer = DocumentIndexer(
            config=IndexerConfig(use_cache=False),
            llm_client=mock_llm,
        )

        # 6 chunks, 4 with distinct section titles (>40% coverage, >=3 distinct)
        chunks = [
            _make_chunk("Part I", 0, 0, "PART I", 1),
            _make_chunk("Item 1", 1, 100, "ITEM 1. BUSINESS", 2),
            _make_chunk("Item 1A", 2, 200, "ITEM 1A. RISK FACTORS", 2),
            _make_chunk("Part II", 3, 300, "PART II", 1),
            _make_chunk("Item 7", 4, 400, "ITEM 7. MD&A", 2),
            _make_chunk("More MD&A", 5, 500, None, 0),
        ]

        result = await indexer._build_structure(chunks, DocumentType.SEC_10K)
        assert len(result) >= 2
        # LLM should NOT have been called
        mock_llm.complete.assert_not_called()
        mock_llm.complete_json.assert_not_called()


# ---------------------------------------------------------------------------
# Step 3: Smarter chunk summary
# ---------------------------------------------------------------------------

class TestChunkSummary:
    def test_shows_all_section_boundaries(self):
        indexer = DocumentIndexer(
            config=IndexerConfig(use_cache=False),
            llm_client=_make_mock_llm(),
        )

        # Create 100 chunks, only some with section titles
        chunks = []
        for i in range(100):
            if i == 0:
                chunks.append(_make_chunk("Part I content", i, i * 100, "PART I", 1))
            elif i == 30:
                chunks.append(_make_chunk("Part II content", i, i * 100, "PART II", 1))
            elif i == 70:
                chunks.append(_make_chunk("Part III content", i, i * 100, "PART III", 1))
            else:
                chunks.append(_make_chunk(f"Content chunk {i}", i, i * 100))

        summary = indexer._build_chunks_summary(chunks)

        # All three section boundaries should be present
        assert "[PART I]" in summary
        assert "[PART II]" in summary
        assert "[PART III]" in summary
        # Should show total count
        assert "Total: 100 chunks" in summary
        # Should not have the old truncation message
        assert "more chunks" not in summary


# ---------------------------------------------------------------------------
# Step 4: Token-aware batching
# ---------------------------------------------------------------------------

class TestTokenAwareBatching:
    def test_respects_token_budget(self):
        config = IndexerConfig(
            summary_token_budget=500,  # Very small budget
            summary_batch_size=10,
            use_cache=False,
        )
        indexer = DocumentIndexer(config=config, llm_client=_make_mock_llm())

        # Create nodes with ~200 chars each (~57 tokens at 3.5 chars/token)
        chunks = ["x" * 200 for _ in range(10)]
        nodes = [
            TreeNode(
                node_id=f"{i:04d}",
                title=f"Section {i}",
                level=0,
                text_span=TextSpan(0, 200, i, i + 1),
            )
            for i in range(10)
        ]

        batches = indexer._create_token_aware_batches(nodes, chunks)

        # With 500 token budget and ~57 tokens per node, should get ~8-9 nodes per batch
        # So 10 nodes should split into 2 batches
        assert len(batches) >= 2
        # All nodes should be included
        total = sum(len(b) for b in batches)
        assert total == 10

    def test_respects_max_batch_size(self):
        config = IndexerConfig(
            summary_token_budget=100000,  # Large budget
            summary_batch_size=3,  # Small batch size cap
            use_cache=False,
        )
        indexer = DocumentIndexer(config=config, llm_client=_make_mock_llm())

        chunks = ["x" * 100 for _ in range(9)]
        nodes = [
            TreeNode(
                node_id=f"{i:04d}",
                title=f"Section {i}",
                level=0,
                text_span=TextSpan(0, 100, i, i + 1),
            )
            for i in range(9)
        ]

        batches = indexer._create_token_aware_batches(nodes, chunks)

        # Should split into 3 batches of 3
        assert len(batches) == 3
        assert all(len(b) == 3 for b in batches)


# ---------------------------------------------------------------------------
# Step 5: Combined structure + summary for small docs
# ---------------------------------------------------------------------------

class TestCombinedStructureSummary:
    @pytest.mark.asyncio
    async def test_small_doc_uses_combined_call(self):
        """For small documents, structure and summaries should be in one LLM call."""
        combined_response = json.dumps([
            {
                "structure": "1",
                "title": "PART I",
                "chunk_index": 0,
                "end_chunk_index": 2,
                "summary": "Part I covers business overview.",
            },
            {
                "structure": "1.1",
                "title": "Item 1. Business",
                "chunk_index": 0,
                "end_chunk_index": 1,
                "summary": "Business section details.",
            },
            {
                "structure": "2",
                "title": "PART II",
                "chunk_index": 2,
                "end_chunk_index": 3,
                "summary": "Part II covers financial data.",
            },
        ])

        mock_llm = _make_mock_llm([combined_response])
        indexer = DocumentIndexer(
            config=IndexerConfig(
                use_cache=False,
                small_doc_threshold=20,
                generate_summaries=True,
            ),
            llm_client=mock_llm,
        )

        chunks = [
            _make_chunk("Business content", 0, 0, "ITEM 1. BUSINESS", 2),
            _make_chunk("More business", 1, 100),
            _make_chunk("Financial data", 2, 200, "ITEM 7. MD&A", 2),
        ]

        result = await indexer._build_structure_with_summaries(chunks, DocumentType.SEC_10K)

        assert len(result) >= 1
        # At least one node should have a summary from the combined call
        all_nodes = []
        indexer._collect_all_nodes(result, all_nodes)
        summaries = [n.summary for n in all_nodes if n.summary]
        assert len(summaries) > 0


# ---------------------------------------------------------------------------
# Step 6: Cache JSON parsing
# ---------------------------------------------------------------------------

class TestCacheJsonParsing:
    @pytest.mark.asyncio
    async def test_caches_parsed_json(self):
        """Verify that _cached_llm_complete_json stores parsed JSON, not raw string."""
        mock_cache = AsyncMock()
        mock_cache.get_llm_response = AsyncMock(return_value=None)
        mock_cache.set_llm_response = AsyncMock()

        mock_llm = _make_mock_llm(['[{"key": "value"}]'])

        indexer = DocumentIndexer(
            config=IndexerConfig(use_cache=True),
            llm_client=mock_llm,
            cache_manager=mock_cache,
        )

        result = await indexer._cached_llm_complete_json("test prompt")
        assert result == [{"key": "value"}]

        # Verify cache was set with JSON string (not raw LLM response)
        mock_cache.set_llm_response.assert_called_once()
        cached_value = mock_cache.set_llm_response.call_args[0][2]
        assert json.loads(cached_value) == [{"key": "value"}]

    @pytest.mark.asyncio
    async def test_cache_hit_returns_parsed_directly(self):
        """On cache hit, should return parsed JSON without calling _extract_json."""
        mock_cache = AsyncMock()
        mock_cache.get_llm_response = AsyncMock(
            return_value=json.dumps([{"cached": True}])
        )

        mock_llm = _make_mock_llm()

        indexer = DocumentIndexer(
            config=IndexerConfig(use_cache=True),
            llm_client=mock_llm,
            cache_manager=mock_cache,
        )

        result = await indexer._cached_llm_complete_json("test prompt")
        assert result == [{"cached": True}]
        mock_llm.complete.assert_not_called()


# ---------------------------------------------------------------------------
# Step 6: Parallel metadata + summaries
# ---------------------------------------------------------------------------

class TestParallelPipeline:
    @pytest.mark.asyncio
    async def test_index_completes_with_all_steps(self):
        """Full integration test that indexing runs through all steps."""
        structure_response = json.dumps([
            {
                "structure": "1",
                "title": "Section 1",
                "chunk_index": 0,
                "end_chunk_index": 1,
                "summary": "Summary of section 1.",
            },
        ])

        mock_llm = _make_mock_llm([structure_response])

        config = IndexerConfig(
            use_cache=False,
            generate_summaries=True,
            extract_metadata=True,
            resolve_cross_refs=True,
            small_doc_threshold=100,  # Use combined path
        )
        indexer = DocumentIndexer(config=config, llm_client=mock_llm)

        doc_index = await indexer.index(
            text="FORM 10-K\n\nPART I\n\nITEM 1. BUSINESS\n\nWe are a company.",
            doc_name="test_10k",
        )

        assert doc_index.doc_name == "test_10k"
        assert len(doc_index.chunks) > 0


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

class TestConvenienceFunction:
    def test_index_document_accepts_summary_model(self):
        """Verify the convenience function accepts summary_model parameter."""
        # Just verify it's callable with the new parameter (actual LLM calls would fail)
        import inspect

        sig = inspect.signature(index_document)
        assert "summary_model" in sig.parameters
        assert sig.parameters["summary_model"].default is None
