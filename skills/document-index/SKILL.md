---
name: "document-index"
description: "Intelligent document processing and querying for financial documents. Use when you need to index, search, or extract information from documents."
---

# DocumentIndex Skills

Comprehensive skills for processing, indexing, searching, and extracting information from structured documents like SEC filings, earnings calls, and research reports.

## Quick Decision Guide

**Choose the right skill based on your needs:**

- **Need to process a new document?** → [document-indexing.md](document-indexing.md)
- **Looking for specific content in a document?** → [node-searching.md](node-searching.md)
- **Have a specific question to answer?** → [agentic-qa.md](agentic-qa.md)
- **Need ALL evidence on a topic for compliance/audit?** → [provenance-extraction.md](provenance-extraction.md)

## Available Patterns

### Core Skills

1. **[Document Indexing](document-indexing.md)** - Transform unstructured documents into hierarchical tree structures with summaries, metadata, and cross-references
2. **[Node Searching](node-searching.md)** - Find relevant document sections using LLM reasoning instead of vector similarity
3. **[Agentic Question Answering](agentic-qa.md)** - Answer questions through iterative reasoning with confidence scoring and full reasoning traces
4. **[Provenance Extraction](provenance-extraction.md)** - Exhaustively scan entire documents to find ALL evidence for topics with progress tracking

## When to Use This Skill

Use DocumentIndex patterns when you need to:
- Process and index financial documents (10-K, 10-Q, 8-K, earnings calls)
- Search for specific information within structured documents
- Answer questions about document content with citations
- Extract comprehensive evidence for compliance, audits, or research
- Navigate complex document hierarchies (PART → ITEM → Section → Note)
- Follow cross-references between document sections

## Complexity Levels

- **Beginner**: Skill 1 (Document Indexing) - Start here to understand the foundation
- **Intermediate**: Skills 2-3 (Node Searching, Agentic QA) - Build on indexed documents
- **Advanced**: Skill 4 (Provenance Extraction) - Comprehensive evidence gathering

## Learning Path

1. Start with **Document Indexing** - Learn how to transform raw documents into queryable structures
2. Explore **Node Searching** - Understand how to find relevant sections using LLM reasoning
3. Try **Agentic QA** - Answer specific questions with confidence scoring and citations
4. Master **Provenance Extraction** - Extract exhaustive evidence for compliance and research

**Recommended progression:**
- First-time users: Read skills in order (1 → 2 → 3 → 4)
- Experienced users: Jump to the skill matching your use case via the Quick Decision Guide

## Typical Workflow

```
1. Index Document (Skill 1)
   ↓
2. Choose your approach:
   ├─ Quick question? → Agentic QA (Skill 3)
   ├─ Find sections? → Node Searching (Skill 2)
   └─ Need ALL evidence? → Provenance Extraction (Skill 4)
```

## Configuration Trade-offs

### Speed vs. Quality
- **Fast/Cheap**: Lower thresholds, fewer iterations, no summaries, multi-model routing
- **Balanced**: Default configurations with multi-model (recommended for most use cases)
- **Thorough**: Higher thresholds, more iterations, with summaries, full excerpt extraction

### Coverage vs. Precision
- **Maximum coverage**: Low relevance thresholds (0.5-0.6)
- **Balanced**: Medium thresholds (0.6-0.7)
- **High precision**: High thresholds (0.7-0.8+)

## Working Examples

See [examples/](examples/) directory for complete, runnable code examples:

- **[indexer_deep_dive.py](examples/indexer_deep_dive.py)** - Document indexing patterns
- **[searcher_showcase.py](examples/searcher_showcase.py)** - Node searching strategies
- **[agentic_qa_tutorial.py](examples/agentic_qa_tutorial.py)** - Question answering workflows
- **[provenance_patterns.py](examples/provenance_patterns.py)** - Evidence extraction patterns
- **[basic_usage.py](examples/basic_usage.py)** - Getting started examples
- **[caching_example.py](examples/caching_example.py)** - Performance optimization
- **[streaming_example.py](examples/streaming_example.py)** - Streaming responses
- **[multi_provider_example.py](examples/multi_provider_example.py)** - Multi-LLM support

See [examples/README.md](examples/README.md) for setup instructions and detailed descriptions.

## Integration Patterns

### Sequential Processing
```
DocumentIndexer → NodeSearcher → AgenticQA
(Index once)   → (Find sections) → (Answer questions)
```

### Parallel Analysis
```
DocumentIndexer → ProvenanceExtractor (multiple topics in parallel)
(Index once)   → (Extract evidence for: climate, cyber, regulatory, etc.)
```

### Hybrid Approach
```
DocumentIndexer → AgenticQA (quick questions)
              ↓
              → ProvenanceExtractor (comprehensive evidence when needed)
```

## Performance Considerations

| Component | Time | Cost | Use Case |
|-----------|------|------|----------|
| Document Indexing | 30-60s | $0.05-0.25 | One-time per document (multi-model) |
| Node Searching | 2-5s | $0.01-0.05 | Per search query (with LLM caching) |
| Agentic QA | 5-15s | $0.02-0.10 | Per question |
| Provenance Extraction | 30-90s | $0.05-0.20 | Per topic (multi-model + batching) |

## Additional Resources

- **Source Code**: `/documentindex` package
- **Documentation**: Project README
- **Main Skills README**: `../README.md`

## Decision Framework Summary

```
┌─────────────────────────────────────────┐
│ What do you need to do?                 │
└─────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
New Doc?    Find Content?  Answer Q?
    │            │            │
    ▼            ▼            ▼
Indexing    Searching      QA
(Skill 1)   (Skill 2)   (Skill 3)
                             │
                    Need ALL evidence?
                             │
                             ▼
                        Provenance
                        (Skill 4)
```

---

**Last Updated**: 2026-02-01
**Skills Version**: 1.0
**Compatible with**: DocumentIndex v0.1.0+
