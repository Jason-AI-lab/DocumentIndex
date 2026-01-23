# DocumentIndex

Lightweight hierarchical tree index for financial documents with reasoning-based retrieval.

## Overview

DocumentIndex builds hierarchical tree structures from financial documents (SEC filings, earnings calls, research reports) and provides two powerful retrieval modes:

- **Agentic QA**: Intelligent, iterative question answering that navigates the document structure
- **Provenance Extraction**: Exhaustive scan to find ALL evidence related to a topic

Unlike vector similarity search, DocumentIndex uses LLM reasoning to understand document structure and find relevant information.

## Features

- 📄 **Hierarchical Tree Indexing**: Understands document structure (PART, ITEM, Note, etc.)
- 🤖 **Multi-Provider LLM Support**: OpenAI, Anthropic, AWS Bedrock, Azure OpenAI, Ollama
- 🔍 **Dual Retrieval Modes**: Agentic QA and Provenance Extraction
- 📊 **Streaming Responses**: Real-time progress tracking and streaming outputs
- 💾 **Caching**: Memory, file, and Redis backends
- 🔗 **Cross-Reference Resolution**: Automatically resolves "see Note 15", "refer to Item 1A"
- 📝 **Metadata Extraction**: Company info, dates, financial numbers

## Installation

```bash
pip install documentindex
```

With Redis support:
```bash
pip install documentindex[cache]
```

## Quick Start

```python
import asyncio
from documentindex import DocumentIndexer, AgenticQA, ProvenanceExtractor

async def main():
    # 1. Index a document
    indexer = DocumentIndexer()
    doc_index = await indexer.index(
        text=your_document_text,
        doc_name="AAPL_10K_2024"
    )
    
    # 2. Ask questions (Agentic QA)
    qa = AgenticQA(doc_index)
    result = await qa.answer("What was the revenue in 2024?")
    print(result.answer)
    print(f"Confidence: {result.confidence}")
    
    # 3. Extract all evidence about a topic (Provenance)
    extractor = ProvenanceExtractor(doc_index)
    evidence = await extractor.extract_all("climate change risks")
    print(f"Found {len(evidence.evidence)} relevant sections")

asyncio.run(main())
```

## Use Cases

### Use Case 1: Document Indexing

```python
from documentindex import DocumentIndexer, IndexerConfig, LLMConfig

# Configure indexer
config = IndexerConfig(
    llm_config=LLMConfig(model="gpt-4o"),
    generate_summaries=True,
    extract_metadata=True,
)

indexer = DocumentIndexer(config)
doc_index = await indexer.index(
    text=document_text,
    doc_name="10K_2024",
)

# Access structure
for node in doc_index.structure:
    print(f"[{node.node_id}] {node.title}")
    for child in node.children:
        print(f"  [{child.node_id}] {child.title}")

# Get text for a node
text = doc_index.get_node_text("0001")
```

### Use Case 2: Question Answering

```python
from documentindex import AgenticQA, AgenticQAConfig

qa = AgenticQA(doc_index)

# Simple question
result = await qa.answer("What are the main risk factors?")

# With configuration
config = AgenticQAConfig(
    max_iterations=5,
    confidence_threshold=0.7,
    follow_cross_refs=True,
)
result = await qa.answer("Explain the revenue breakdown", config)

# Access citations
for citation in result.citations:
    print(f"- {citation.node_title}: {citation.excerpt}")
```

### Use Case 3: Streaming Responses

```python
# Stream QA answer
result = await qa.answer_stream("What was the revenue?")
async for chunk in result.answer_stream:
    print(chunk.content, end="", flush=True)

# Progress callbacks
def on_progress(update):
    print(f"[{update.progress_pct:.1f}%] {update.step_name}")

doc_index = await indexer.index_with_progress(
    text=document_text,
    progress_callback=on_progress,
)
```

### Use Case 4: Provenance Extraction

```python
from documentindex import ProvenanceExtractor, ProvenanceConfig

extractor = ProvenanceExtractor(doc_index)

# Extract evidence for single topic
result = await extractor.extract_all(
    topic="environmental sustainability",
    config=ProvenanceConfig(
        relevance_threshold=0.6,
        extract_excerpts=True,
    ),
)

# Multiple topics
topics = {
    "climate": "climate change and environmental risks",
    "regulatory": "regulatory compliance requirements",
    "financial": "revenue and financial performance",
}
results = await extractor.extract_by_category(topics)
```

## LLM Provider Configuration

### OpenAI

```python
from documentindex import create_openai_client, LLMConfig

# Using factory
client = create_openai_client(model="gpt-4o")

# Using config
config = LLMConfig(model="gpt-4o")
# Or: model="openai/gpt-4-turbo"
```

### Anthropic

```python
from documentindex import create_anthropic_client

client = create_anthropic_client(model="claude-sonnet-4-20250514")

# Or via config
config = LLMConfig(model="anthropic/claude-sonnet-4-20250514")
```

### AWS Bedrock

```python
from documentindex import create_bedrock_client

client = create_bedrock_client(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    region="us-east-1",
)

# Or via config
config = LLMConfig(
    model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    provider_config={"aws_region_name": "us-east-1"},
)
```

### Azure OpenAI

```python
from documentindex import create_azure_client

client = create_azure_client(
    deployment_name="gpt-4",
    api_base="https://your-resource.openai.azure.com",
    api_version="2024-02-15-preview",
)
```

### Local (Ollama)

```python
from documentindex import create_ollama_client

client = create_ollama_client(
    model="llama2",
    base_url="http://localhost:11434",
)
```

## Caching

### Memory Cache (Development)

```python
from documentindex import CacheConfig, CacheManager

config = CacheConfig(backend="memory", memory_max_size=1000)
cache = CacheManager(config)
```

### File Cache (Persistence)

```python
config = CacheConfig(
    backend="file",
    file_cache_dir=".cache/documentindex",
)
cache = CacheManager(config)
```

### Redis Cache (Production)

```python
config = CacheConfig(
    backend="redis",
    redis_host="localhost",
    redis_port=6379,
)
cache = CacheManager(config)
```

### Using Cache with Components

```python
indexer = DocumentIndexer(config, cache_manager=cache)
searcher = NodeSearcher(doc_index, cache_manager=cache)
```

## Supported Document Types

DocumentIndex automatically detects document types:

- **SEC Filings**: 10-K, 10-Q, 8-K, DEF 14A, S-1, 20-F, 6-K
- **Earnings Documents**: Earnings calls, earnings releases
- **Analysis**: Research reports, press releases
- **Generic**: Any text document

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `DocumentIndexer` | Builds hierarchical tree from text |
| `DocumentIndex` | Container for indexed document |
| `NodeSearcher` | Searches for related nodes |
| `AgenticQA` | Question answering with reasoning |
| `ProvenanceExtractor` | Exhaustive evidence extraction |

### Data Models

| Model | Description |
|-------|-------------|
| `TreeNode` | Node in document tree |
| `TextSpan` | Maps to original text |
| `NodeMatch` | Search result with relevance |
| `Citation` | Citation to document location |
| `QAResult` | Question answering result |
| `ProvenanceResult` | Provenance extraction result |

### Configuration Classes

| Config | Description |
|--------|-------------|
| `LLMConfig` | LLM provider settings |
| `IndexerConfig` | Indexing options |
| `AgenticQAConfig` | QA behavior settings |
| `ProvenanceConfig` | Extraction settings |
| `CacheConfig` | Cache backend settings |

## Examples

See the `examples/` directory for complete examples:

- `basic_usage.py` - Getting started
- `streaming_example.py` - Progress and streaming
- `multi_provider_example.py` - Different LLM providers
- `caching_example.py` - Cache configurations

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=documentindex

# Format code
ruff format src tests

# Lint code
ruff check src tests
```

## License

MIT License

## Contributing

Contributions are welcome! Please read our contributing guidelines first.
