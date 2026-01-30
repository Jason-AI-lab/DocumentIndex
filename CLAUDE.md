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

1. **DocumentIndexer** (`src/documentindex/indexer.py`): Builds a hierarchical tree from documents, understanding structure like PART/ITEM/Note hierarchies in financial documents.

2. **AgenticQA** (`src/documentindex/agentic_qa.py`): Implements iterative question-answering by reasoning through the document tree, following cross-references, and building confidence in answers.

3. **ProvenanceExtractor** (`src/documentindex/provenance.py`): Performs exhaustive searches across the entire document tree to find all evidence related to specified topics.

4. **NodeSearcher** (`src/documentindex/search.py`): Core search functionality that uses LLM reasoning to find relevant nodes based on queries.

### Key Design Patterns

- **Async-First**: All core operations are async using `asyncio`
- **LLM Provider Abstraction**: Uses `litellm` for multi-provider support (OpenAI, Anthropic, Bedrock, Azure, Ollama)
- **Streaming Support**: Built-in streaming for real-time progress and responses
- **Caching Strategy**: Three-tier caching (memory, file, Redis) with consistent interface

### Data Flow

1. **Indexing**: Raw text → Document detection → Chunking → Tree building → Node summaries
2. **QA**: Question → Initial search → Follow cross-refs → Build answer → Confidence scoring
3. **Provenance**: Topic → Parallel node evaluation → Evidence collection → Categorization

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
- `IndexerConfig`: Indexing behavior
- `AgenticQAConfig`: QA iterations and confidence thresholds
- `ProvenanceConfig`: Evidence extraction settings
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