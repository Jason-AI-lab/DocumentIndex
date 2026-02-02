# DocumentIndex Examples

Working code examples demonstrating each DocumentIndex skill pattern.

## Quick Start

### Installation

```bash
# Install DocumentIndex
pip install documentindex

# Or install from source
git clone https://github.com/yourusername/DocumentIndex.git
cd DocumentIndex
pip install -e .
```

### Running Examples

```bash
# Navigate to examples directory
cd skills/document-index/examples

# Run individual examples
python indexer_deep_dive.py
python searcher_showcase.py
python agentic_qa_tutorial.py
python provenance_patterns.py
```

## Examples Overview

| Example | Skill | Description | Complexity |
|---------|-------|-------------|------------|
| [basic_usage.py](basic_usage.py) | All | Getting started with all components | ⭐ Beginner |
| [indexer_deep_dive.py](indexer_deep_dive.py) | [Document Indexing](../document-indexing.md) | Comprehensive indexing patterns | ⭐⭐ Intermediate |
| [searcher_showcase.py](searcher_showcase.py) | [Node Searching](../node-searching.md) | Search strategies and optimization | ⭐⭐ Intermediate |
| [agentic_qa_tutorial.py](agentic_qa_tutorial.py) | [Agentic QA](../agentic-qa.md) | Question answering workflows | ⭐⭐ Intermediate |
| [provenance_patterns.py](provenance_patterns.py) | [Provenance Extraction](../provenance-extraction.md) | Evidence extraction and reporting | ⭐⭐⭐ Advanced |
| [caching_example.py](caching_example.py) | All | Performance optimization with caching | ⭐⭐ Intermediate |
| [streaming_example.py](streaming_example.py) | QA, Provenance | Streaming responses for real-time UX | ⭐⭐ Intermediate |
| [multi_provider_example.py](multi_provider_example.py) | All | Using multiple LLM providers | ⭐⭐ Intermediate |

## What Each Example Demonstrates

### Example 1: Basic Usage (⭐ Beginner)
**File**: [basic_usage.py](basic_usage.py)

Demonstrates:
- Quick start with minimal configuration
- Basic indexing, searching, and question answering
- Simple end-to-end workflow
- Default settings and best practices

**Best for**: First-time users, quick prototypes

---

### Example 2: Indexer Deep Dive (⭐⭐ Intermediate)
**File**: [indexer_deep_dive.py](indexer_deep_dive.py) | **Skill**: [Document Indexing](../document-indexing.md)

Demonstrates:
- Structure detection for different document types (10-K, 10-Q, 8-K)
- Node summarization and metadata extraction
- Cross-reference resolution
- Custom chunking strategies
- Performance optimization

**Best for**: Understanding document indexing in depth

---

### Example 3: Searcher Showcase (⭐⭐ Intermediate)
**File**: [searcher_showcase.py](searcher_showcase.py) | **Skill**: [Node Searching](../node-searching.md)

Demonstrates:
- High precision vs. comprehensive search strategies
- Relevance threshold tuning
- Batch processing and caching
- Multi-query relevance patterns
- Progressive refinement techniques

**Best for**: Optimizing search performance and accuracy

---

### Example 4: Agentic QA Tutorial (⭐⭐ Intermediate)
**File**: [agentic_qa_tutorial.py](agentic_qa_tutorial.py) | **Skill**: [Agentic QA](../agentic-qa.md)

Demonstrates:
- Quick questions vs. thorough analysis
- Confidence score interpretation
- Multi-hop reasoning for complex questions
- Batch question processing
- Interactive Q&A patterns
- Reasoning trace analysis

**Best for**: Building question-answering applications

---

### Example 5: Provenance Patterns (⭐⭐⭐ Advanced)
**File**: [provenance_patterns.py](provenance_patterns.py) | **Skill**: [Provenance Extraction](../provenance-extraction.md)

Demonstrates:
- Single-topic exhaustive extraction
- Multi-category parallel analysis
- Threshold tuning for precision/recall
- Summary generation
- Progress tracking for long operations
- Streaming extraction
- Export formats (JSON, CSV, Markdown)

**Best for**: Compliance, audits, comprehensive research

---

### Example 6: Caching Example (⭐⭐ Intermediate)
**File**: [caching_example.py](caching_example.py)

Demonstrates:
- Result caching for repeated queries
- Cache invalidation strategies
- Performance benchmarking
- Cost optimization

**Best for**: Production deployments, cost optimization

---

### Example 7: Streaming Example (⭐⭐ Intermediate)
**File**: [streaming_example.py](streaming_example.py)

Demonstrates:
- Streaming QA responses for real-time UX
- Streaming provenance extraction
- Progress callbacks
- Async/await patterns

**Best for**: Interactive applications, user-facing tools

---

### Example 8: Multi-Provider Example (⭐⭐ Intermediate)
**File**: [multi_provider_example.py](multi_provider_example.py)

Demonstrates:
- Using OpenAI, Anthropic, and other LLM providers
- Provider-specific configurations
- Fallback strategies
- Cost comparison across providers

**Best for**: Multi-cloud deployments, provider flexibility

## Common Patterns

### Pattern 1: Index Once, Query Many
```python
# Index document once
indexer = DocumentIndexer()
doc_index = await indexer.index(text=document_text, doc_name="AAPL_10K")

# Use for multiple queries
searcher = NodeSearcher(doc_index, llm_client)
qa = AgenticQA(doc_index, llm_client)
extractor = ProvenanceExtractor(doc_index, llm_client)

# Run multiple queries
matches = await searcher.find_related_nodes("revenue")
answer = await qa.answer("What was the revenue?")
evidence = await extractor.extract_all("climate risks")
```

### Pattern 2: Progressive Refinement
```python
# Start with broad search
broad_matches = await searcher.find_related_nodes("risks")

# Refine with specific question
specific_answer = await qa.answer("What are the cybersecurity risks?")

# Get exhaustive evidence if needed
all_evidence = await extractor.extract_all("cybersecurity risks")
```

### Pattern 3: Multi-Category Analysis
```python
# Extract evidence for multiple topics in parallel
categories = {
    "climate": "climate change and environmental risks",
    "cyber": "cybersecurity and data protection",
    "regulatory": "regulatory compliance requirements"
}
results = await extractor.extract_by_category(categories)

# Compare coverage
for category, result in results.items():
    print(f"{category}: {len(result.evidence)} sections")
```

## Customizing for Your Use Case

### For SEC Filings
```python
# Optimize for 10-K/10-Q structure
config = IndexerConfig(
    generate_summaries=True,      # Critical for navigation
    extract_metadata=True,         # Get dates, numbers, entities
    resolve_cross_refs=True,       # Follow "See Note X"
    chunk_config=ChunkConfig(max_chunk_tokens=1000)
)
```

### For Earnings Calls
```python
# Optimize for transcript structure
config = IndexerConfig(
    generate_summaries=True,
    extract_metadata=False,        # Less structured metadata
    resolve_cross_refs=False,      # Fewer cross-references
    chunk_config=ChunkConfig(max_chunk_tokens=1500)
)
```

### For Research Reports
```python
# Optimize for narrative structure
config = IndexerConfig(
    generate_summaries=True,
    extract_metadata=True,
    resolve_cross_refs=True,
    chunk_config=ChunkConfig(max_chunk_tokens=2000)
)
```

## Troubleshooting

### Issue: Import Errors
**Solution:**
```bash
# Ensure DocumentIndex is installed
pip install documentindex

# Or install in development mode
pip install -e /path/to/DocumentIndex
```

### Issue: API Rate Limits
**Solution:**
```python
# Reduce concurrency
config = IndexerConfig(max_concurrent_summaries=2)

# Use caching
config = NodeSearchConfig(use_cache=True)

# Increase batch size
config = ProvenanceConfig(batch_size=15)
```

### Issue: Slow Performance
**Solution:**
```python
# Disable summaries for speed
config = IndexerConfig(generate_summaries=False)

# Lower relevance threshold
config = NodeSearchConfig(relevance_threshold=0.5)

# Use streaming for large results
async for match in extractor.extract_stream(topic):
    process(match)
```

### Issue: Low Quality Results
**Solution:**
```python
# Enable summaries
config = IndexerConfig(generate_summaries=True)

# Increase relevance threshold
config = NodeSearchConfig(relevance_threshold=0.7)

# Increase QA iterations
config = AgenticQAConfig(max_iterations=10)
```

## Environment Variables

```bash
# OpenAI API Key
export OPENAI_API_KEY="your-key-here"

# Anthropic API Key
export ANTHROPIC_API_KEY="your-key-here"

# Optional: Set default model
export DOCUMENTINDEX_DEFAULT_MODEL="gpt-4"
```

## Additional Resources

- **Main Skills Directory**: [../](../)
- **Project Documentation**: [../../../README.md](../../../README.md)
- **API Reference**: [Documentation link]
- **GitHub Issues**: [Issues link]

## Contributing Examples

To contribute a new example:

1. Follow the existing file structure
2. Include comprehensive docstrings
3. Add error handling
4. Update this README with your example
5. Test with multiple document types

---

**Last Updated**: 2026-02-01
**Examples Version**: 1.0
**Compatible with**: DocumentIndex v0.1.0+
