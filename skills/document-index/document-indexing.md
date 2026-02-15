---
name: "document-indexing"
description: "Transform unstructured documents into hierarchical tree structures. Use when you need to index a new document for querying."
---

# Skill: Document Indexing with DocumentIndexer

## Overview
Build hierarchical tree structures from financial documents (SEC filings, earnings calls, research reports) that understand document structure (PART, ITEM, Note, etc.).

## Core Capability
Transform unstructured text into a navigable tree index with automatic structure detection, node summarization, metadata extraction, and cross-reference resolution.

## API Pattern

```python
from documentindex import DocumentIndexer, IndexerConfig, ChunkConfig, LLMConfig

# Basic usage
indexer = DocumentIndexer()
doc_index = await indexer.index(
    text=document_text,
    doc_name="AAPL_10K_2024"
)

# With configuration (including multi-model support)
config = IndexerConfig(
    llm_config=LLMConfig(model="gpt-4o"),              # Structure detection
    summary_llm_config=LLMConfig(model="gpt-4o-mini"),  # Cheaper model for summaries
    chunk_config=ChunkConfig(max_chunk_tokens=1000),
    generate_summaries=True,      # Generate node summaries
    extract_metadata=True,         # Extract entities, dates, numbers
    resolve_cross_refs=True,       # Resolve "See Note 15" references
    max_concurrent_summaries=5,    # Parallel processing limit
    summary_batch_size=10,         # Nodes per batched summary call
    summary_token_budget=8000,     # Max input tokens per summary batch
)
indexer = DocumentIndexer(config)
doc_index = await indexer.index(text=document_text, doc_name="doc_name")

# Convenience function with multi-model
from documentindex import index_document
doc_index = await index_document(
    text=document_text,
    doc_name="doc_name",
    model="gpt-4o",
    summary_model="gpt-4o-mini",
)
```

## Key Features

### 1. Automatic Structure Detection

- Detects document type (10-K, 10-Q, 8-K, earnings calls, etc.)
- Identifies hierarchical structure (PART → ITEM → Section → Note)
- Skips LLM when chunk metadata has clear section headers (LLM-skip optimization)
- Creates tree nodes with parent-child relationships

### 2. Node Summarization

- Generates concise summaries using token-aware batched LLM calls
- Leaf nodes summarized from raw text, parent nodes synthesized from children (bottom-up)
- Supports separate cheaper LLM model via `summary_llm_config`
- Small documents use a combined structure+summary call for efficiency
- Configurable via `generate_summaries=True`

### 3. Metadata Extraction
- Company names, dates, financial figures
- Entities and key metrics
- Configurable via `extract_metadata=True`

### 4. Cross-Reference Resolution
- Detects references like "See Note 15", "refer to Item 1A"
- Creates navigable links between nodes
- Configurable via `resolve_cross_refs=True`

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `llm_config` | LLMConfig | Required | LLM for structure detection |
| `summary_llm_config` | LLMConfig | None | Cheaper LLM for summaries (defaults to llm_config) |
| `chunk_config` | ChunkConfig | Default | Chunking parameters |
| `generate_summaries` | bool | True | Generate node summaries |
| `extract_metadata` | bool | True | Extract structured metadata |
| `resolve_cross_refs` | bool | True | Resolve cross-references |
| `max_concurrent_summaries` | int | 5 | Parallel summary generation limit |
| `summary_batch_size` | int | 10 | Nodes per batched summary call |
| `summary_token_budget` | int | 8000 | Max input tokens per summary batch |
| `max_concurrent_batches` | int | 3 | Parallel batch processing limit |
| `small_doc_threshold` | int | 15 | Chunks - docs smaller use combined structure+summary |
| `use_cache` | bool | True | Cache LLM responses |

## Output Structure

```python
# DocumentIndex contains:
doc_index.doc_id           # Unique document identifier
doc_index.doc_name         # Document name
doc_index.structure        # List of root TreeNode objects
doc_index.metadata         # DocumentMetadata (if extracted)

# TreeNode hierarchy:
node.node_id              # Unique ID (e.g., "0001")
node.title                # Section title
node.summary              # Generated summary (if enabled)
node.children             # List of child nodes
node.text_span            # TextSpan (start/end positions)
node.cross_references     # List of CrossReference objects

# Access methods:
doc_index.find_node(node_id)           # Find node by ID
doc_index.get_node_text(node_id)       # Get original text
doc_index.get_all_nodes()              # Flatten tree to list
```

## Common Use Cases

### Use Case 1: Quick Indexing
```python
# Minimal configuration for fast indexing
config = IndexerConfig(
    llm_config=llm_client.config,
    generate_summaries=False,
    extract_metadata=False,
    max_concurrent_summaries=5
)
```

### Use Case 2: Full Analysis
```python
# Maximum detail for comprehensive analysis
config = IndexerConfig(
    llm_config=llm_client.config,
    generate_summaries=True,
    extract_metadata=True,
    resolve_cross_refs=True
)
```

### Use Case 3: Custom Chunking
```python
# Optimize for specific document characteristics
config = IndexerConfig(
    llm_config=llm_client.config,
    chunk_config=ChunkConfig(
        max_chunk_tokens=2000,
        overlap_tokens=200,
        min_chunk_tokens=100
    )
)
```

## Best Practices

1. **Enable summaries for navigation**: Summaries dramatically improve search relevance
2. **Use multi-model for cost savings**: Set `summary_llm_config` to a cheaper model (e.g., gpt-4o-mini) for summaries
3. **Use metadata extraction for structured queries**: Enables finding specific dates, numbers, entities
4. **Resolve cross-references for comprehensive retrieval**: Critical for following document links
5. **Adjust concurrency based on rate limits**: Higher concurrency = faster but may hit API limits
6. **Enable caching**: LLM response caching avoids redundant calls on re-indexing

## Performance Considerations

- **Indexing time**: ~30-60 seconds for typical 10-K (50-100 pages) with summaries
- **API costs**: ~$0.05-0.25 per document with multi-model (cheap summary model saves 50%+)
- **LLM-skip**: Well-sectioned documents (e.g., SEC filings) skip LLM structure detection entirely
- **Memory usage**: ~10-50MB per indexed document in memory
- **Recommended settings for production**:
  - `generate_summaries=True` (critical for search quality)
  - `summary_llm_config=LLMConfig(model="gpt-4o-mini")` (cost optimization)
  - `max_concurrent_summaries=5` (balance speed and rate limits)
  - `summary_token_budget=8000` (optimal batch size for summary calls)
  - `use_cache=True` (essential for re-indexing performance)

## Example Output

```
Document Structure:
[0000] PART I
  [0001] ITEM 1. BUSINESS
    Summary: Apple designs, manufactures and markets smartphones...
    Metadata: {company: "Apple Inc.", fiscal_year: 2024}
  [0002] ITEM 1A. RISK FACTORS
    Cross-refs: ["See Note 15", "Refer to Item 7"]
[0010] PART II
  [0011] ITEM 7. MANAGEMENT'S DISCUSSION
```

## Integration with Other Components

Once indexed, use the `DocumentIndex` object with:
- **NodeSearcher**: Find relevant sections by query
- **AgenticQA**: Answer questions about the document
- **ProvenanceExtractor**: Extract all evidence about a topic

## Example Code

See [examples/indexer_deep_dive.py](examples/indexer_deep_dive.py) for a complete, runnable example demonstrating:
- Basic and advanced indexing configurations
- Structure detection and navigation
- Metadata extraction and cross-reference resolution
- Performance optimization strategies

## Reference
See `examples/indexer_deep_dive.py` for comprehensive demonstrations.
