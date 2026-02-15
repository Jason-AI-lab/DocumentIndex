# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DocumentIndex is a lightweight hierarchical tree index for financial documents with reasoning-based retrieval. It uses LLM reasoning to navigate document structure rather than relying on vector similarity search.

Key capabilities:
- **Agentic QA**: Intelligent, iterative question answering that navigates the document tree
- **Provenance Extraction**: Exhaustive scan to find ALL evidence related to a topic

## Development Commands

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with coverage
pytest --cov=documentindex

# Run a specific test file
pytest tests/test_cache.py

# Run a specific test
pytest tests/test_cache.py::test_memory_cache_operations

# Format code
ruff format src tests

# Lint code
ruff check src tests
```

## Architecture

### Core Components

The system is built around a hierarchical tree indexing approach:

1. **DocumentIndexer** (`src/documentindex/indexer.py`): Builds a hierarchical tree from documents. Supports multi-model (separate summary LLM), LLM-skip for well-sectioned documents, token-aware batched summaries with bottom-up parent synthesis, and combined structure+summary calls for small documents.

2. **AgenticQA** (`src/documentindex/agentic_qa.py`): Implements iterative question-answering by reasoning through the document tree, following cross-references, and building confidence in answers.

3. **ProvenanceExtractor** (`src/documentindex/provenance.py`): Exhaustive evidence extraction with multi-model support (cheap model for scoring/summary, capable model for excerpts), token-aware batched excerpt extraction using full node content, excerpt threshold filtering, and LLM response caching.

4. **NodeSearcher** (`src/documentindex/searcher.py`): Core search functionality with LLM-level response caching, configurable concurrency, and batched cross-reference scoring.

### Key Design Patterns

- **Async-First**: All core operations are async using `asyncio`
- **LLM Provider Abstraction**: Uses `litellm` for multi-provider support (OpenAI, Anthropic, Bedrock, Azure, Ollama)
- **Multi-Model Routing**: Separate LLM configs for different task types (e.g., cheap model for scoring/summaries, capable model for structure detection and excerpt extraction)
- **Streaming Support**: Built-in streaming for real-time progress and responses
- **Caching Strategy**: Three-tier caching (memory, file, Redis) with LLM-level response caching across indexer, searcher, and provenance
- **Token-Aware Batching**: Intelligent grouping of LLM calls by estimated token budget to minimize API round-trips

### Data Flow

1. **Indexing**: Raw text → Document detection → Chunking → Structure detection (LLM or chunk metadata skip) → Batched summaries (leaf nodes from text, parents synthesized from children) → Metadata + cross-refs in parallel
2. **QA**: Question → Initial search → Follow cross-refs → Build answer → Confidence scoring
3. **Provenance**: Topic → Batched node scoring (cached) → Token-aware batched excerpt extraction (full node content, no truncation) → Summary generation

## Testing Approach

- Uses `pytest` with `pytest-asyncio` for async test support
- Mock LLM clients in `conftest.py` for consistent testing
- Test files mirror source structure (e.g., `test_cache.py` for `cache.py`)
- Coverage tracking with `pytest-cov`

## Important Implementation Details

### Cross-Reference Resolution
The system automatically detects and follows references like "see Note 15" or "refer to Item 1A" using regex patterns in `src/documentindex/cross_ref.py`.

### Document Type Detection
Automatic detection of SEC filings (10-K, 10-Q, etc.) and other financial documents in `src/documentindex/detector.py` to apply appropriate parsing strategies.

### Metadata Extraction
Extracts structured information (company names, dates, financial figures) during indexing via `src/documentindex/metadata.py`.

### Configuration System
Extensive configuration options through dataclasses:
- `LLMConfig`: Provider settings and model selection
- `IndexerConfig`: Indexing behavior (includes `summary_llm_config` for multi-model, `summary_token_budget`, `summary_batch_size`, `small_doc_threshold`)
- `AgenticQAConfig`: QA iterations and confidence thresholds
- `ProvenanceConfig`: Evidence extraction settings (includes `scoring_llm_config` for multi-model, `excerpt_threshold`, `excerpt_token_budget`, `max_concurrent_categories`)
- `NodeSearchConfig`: Search behavior (includes `batch_size`, `max_concurrent_batches`)
- `CacheConfig`: Cache backend selection

## Common Development Tasks

### Adding a New LLM Provider
1. Create client in `src/documentindex/llm_client.py`
2. Add factory function following pattern of `create_openai_client()`
3. Update `LLMConfig` model parsing in `models.py`

### Implementing a New Retrieval Mode
1. Create new class inheriting from base patterns in `agentic_qa.py` or `provenance.py`
2. Implement async methods for search logic
3. Add streaming support if applicable
4. Create corresponding config class in `models.py`

### Testing New Components
1. Add test file in `tests/` directory
2. Use `MockLLMClient` from `conftest.py` for LLM operations
3. Test both sync and async operations
4. Include edge cases for error handling