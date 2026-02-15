# DocumentIndex Skills

This directory contains skill documents that summarize the core capabilities of DocumentIndex components. These markdown files serve as:

1. **LLM Training Material**: Structured knowledge for training language models on DocumentIndex usage
2. **Quick Reference Guides**: Concise API patterns and best practices
3. **Integration Templates**: Copy-paste examples for common use cases
4. **Decision Frameworks**: Guidelines for choosing configurations and approaches

## Available Skills

### 🏗️ [Document Indexing](./document-indexing.md)
**Core Capability**: Transform unstructured text into hierarchical tree structures with summaries, metadata, and cross-references.

**Key Topics**:
- Automatic structure detection with LLM-skip for well-sectioned documents
- Node summarization with token-aware batching and multi-model support
- Metadata extraction (dates, numbers, entities)
- Cross-reference resolution ("See Note 15")
- Multi-model routing (cheap model for summaries, capable for structure)
- LLM response caching and performance tuning

**Use When**: You need to index a new document for querying

---

### 🔍 [Node Searching](./node-searching.md)
**Core Capability**: Find relevant document sections using LLM reasoning instead of vector similarity.

**Key Topics**:
- Relevance scoring (0.0-1.0) with reasoning
- Cross-reference expansion with batched scoring
- Batch processing with configurable concurrency
- LLM-level response caching (shared across components)
- Threshold tuning for precision/recall

**Use When**: You need to find sections related to a query

---

### 🤖 [Agentic Question Answering](./agentic-qa.md)
**Core Capability**: Answer questions through iterative reasoning with confidence scoring and full reasoning traces.

**Key Topics**:
- Multi-step reasoning (plan → search → read → synthesize)
- Confidence-based stopping
- Citation generation
- Multi-hop reasoning for complex questions
- Streaming responses
- Reasoning trace analysis

**Use When**: You need direct answers to specific questions

---

### 📊 [Provenance Extraction](./provenance-extraction.md)
**Core Capability**: Exhaustively scan entire documents to find ALL evidence for topics with progress tracking.

**Key Topics**:
- 100% document coverage guarantee
- Multi-model routing (cheap scoring, capable excerpts)
- Token-aware batched excerpt extraction with full content
- Multi-category parallel extraction with scoring cache
- Excerpt threshold for cost optimization
- LLM response caching across extractions
- Summary generation, progress tracking, streaming results
- Export formats (JSON, CSV, Markdown)

**Use When**: You need comprehensive evidence for compliance, audits, or research

---

## Skill File Structure

Each skill document follows this structure:

```markdown
# Skill: [Component Name]

## Overview
One-sentence capability summary

## Core Capability
Detailed explanation of what it does

## API Pattern
Basic and advanced usage examples

## Key Features
Numbered list of main capabilities

## Configuration Options
Table of all configuration parameters

## Output Structure
Description of return values and data structures

## Common Use Cases
3-4 real-world scenarios with code

## Best Practices
Guidelines for optimal usage

## Performance Considerations
Timing, costs, and recommended settings

## Example Output
Sample output showing what to expect

## Integration with Other Components
How it fits in the larger system

## Advanced Patterns
Complex usage scenarios

## Reference
Link to comprehensive example file
```

## Usage Patterns

### For LLM Training
Use these documents to train language models on DocumentIndex capabilities:

```python
# Load skill documents
with open("skills/document-indexing.md") as f:
    indexing_knowledge = f.read()

# Use in prompts
prompt = f"""
You are an expert in document indexing using the DocumentIndexer component.

{indexing_knowledge}

User question: How do I index a 10-K filing with metadata extraction?
"""
```

### For Quick Reference
Jump to the relevant skill when you need to:

- **Index a document** → `document-indexing.md`
- **Search for content** → `node-searching.md`
- **Answer questions** → `agentic-qa.md`
- **Extract all evidence** → `provenance-extraction.md`

### For Integration
Copy code examples from "API Pattern" and "Common Use Cases" sections for rapid prototyping.

## Decision Framework

### Which component should I use?

```
Need to process a new document?
└─> Use DocumentIndexer (document-indexing.md)

Have a specific question?
└─> Use AgenticQA (agentic-qa.md)

Need to find relevant sections?
└─> Use NodeSearcher (node-searching.md)

Need ALL evidence on a topic?
└─> Use ProvenanceExtractor (provenance-extraction.md)
```

### Which configuration should I use?

**Speed vs. Quality Trade-offs**:
- Fast/Cheap: Lower thresholds, fewer iterations, no summaries
- Balanced: Default configurations (recommended for most use cases)
- Thorough: Higher thresholds, more iterations, with summaries

**Coverage vs. Precision**:
- Maximum coverage: Low relevance thresholds (0.5-0.6)
- Balanced: Medium thresholds (0.6-0.7)
- High precision: High thresholds (0.7-0.8+)

## Comprehensive Examples

Each skill document references a comprehensive example file:

- `document-indexing.md` → `examples/indexer_deep_dive.py`
- `node-searching.md` → `examples/searcher_showcase.py`
- `agentic-qa.md` → `examples/agentic_qa_tutorial.py`
- `provenance-extraction.md` → `examples/provenance_patterns.py`

## Contributing

When adding new skills:
1. Follow the existing structure template
2. Include code examples for all API patterns
3. Provide real-world use cases
4. Add performance considerations
5. Link to comprehensive example files

## License

MIT License - Same as DocumentIndex project
