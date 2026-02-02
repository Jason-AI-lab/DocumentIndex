---
name: "node-searching"
description: "Find relevant document sections using LLM reasoning. Use when you need to locate specific content within an indexed document."
---

# Skill: Node Searching with NodeSearcher

## Overview
Search document tree for nodes related to queries using LLM reasoning to evaluate relevance, not vector similarity.

## Core Capability
Find all nodes in a document tree that are relevant to a query, with relevance scores, match reasoning, and optional cross-reference expansion.

## API Pattern

```python
from documentindex import NodeSearcher, NodeSearchConfig

# Basic usage
searcher = NodeSearcher(doc_index, llm_client=llm_client)
matches = await searcher.find_related_nodes("revenue growth")

# With configuration
config = NodeSearchConfig(
    relevance_threshold=0.7,    # Min score to include (0-1)
    max_results=20,             # Max nodes to return
    include_children=True,      # Include child nodes
    follow_cross_refs=True,     # Follow cross-references
    use_cache=True,             # Cache results
    batch_size=10               # Nodes to score per LLM call
)
matches = await searcher.find_related_nodes("revenue growth", config)
```

## Key Features

### 1. LLM-Based Relevance Scoring
- Each node scored 0.0-1.0 based on query relevance
- Includes reasoning for each score
- More accurate than vector similarity for structured documents

### 2. Match Reasoning
- Explains why each node is relevant
- Helps understand search results
- Useful for debugging and validation

### 3. Cross-Reference Expansion
- Follows "See Note 15" style references
- Expands search to related sections
- Configurable via `follow_cross_refs=True`

### 4. Batch Processing
- Processes multiple nodes per LLM call
- Reduces API costs and latency
- Configurable via `batch_size`

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `relevance_threshold` | float | 0.5 | Minimum relevance score (0-1) |
| `max_results` | int | 20 | Maximum nodes to return |
| `include_children` | bool | True | Include child nodes of matches |
| `follow_cross_refs` | bool | True | Follow cross-references |
| `use_cache` | bool | True | Enable result caching |
| `batch_size` | int | 10 | Nodes to score per LLM call |

## Output Structure

```python
# Returns list of NodeMatch objects:
match.node                # TreeNode object
match.relevance_score     # Float 0.0-1.0
match.match_reason        # String explanation
match.excerpts            # List of relevant text excerpts
match.matched_via_ref     # True if found via cross-reference
```

## Relevance Score Interpretation

| Score Range | Meaning | Use Case |
|-------------|---------|----------|
| 0.9-1.0 | Highly relevant | Direct answer to query |
| 0.8-0.9 | Very relevant | Contains key information |
| 0.7-0.8 | Relevant | Related context |
| 0.6-0.7 | Somewhat relevant | Background information |
| 0.5-0.6 | Marginally relevant | Tangential mention |
| < 0.5 | Not relevant | Filtered out by default |

## Common Use Cases

### Use Case 1: High Precision Search
```python
# Find only the most relevant sections
config = NodeSearchConfig(
    relevance_threshold=0.8,
    max_results=5,
    follow_cross_refs=False
)
matches = await searcher.find_related_nodes("cybersecurity risks", config)
```

### Use Case 2: Comprehensive Search
```python
# Cast a wide net to find all related content
config = NodeSearchConfig(
    relevance_threshold=0.5,
    max_results=50,
    follow_cross_refs=True
)
matches = await searcher.find_related_nodes("climate change", config)
```

### Use Case 3: Batch Queries
```python
# Search multiple topics efficiently
queries = ["revenue", "risks", "competition"]
for query in queries:
    matches = await searcher.find_related_nodes(query)
    print(f"{query}: {len(matches)} matches")
```

## Best Practices

1. **Start with default threshold (0.5)**: Adjust based on results
2. **Enable cross-reference following**: Critical for comprehensive retrieval
3. **Use caching for repeated queries**: Dramatically speeds up repeated searches
4. **Batch size 10-15 is optimal**: Balance between API efficiency and latency
5. **Review match reasoning**: Helps validate search quality

## Performance Considerations

- **Search time**: ~2-5 seconds for typical 30-40 node document
- **API costs**: ~$0.01-0.05 per search depending on document size
- **Caching effectiveness**: 100% speedup for repeated queries
- **Recommended settings for production**:
  - `relevance_threshold=0.6` (good precision/recall balance)
  - `batch_size=10` (standard efficiency)
  - `use_cache=True` (essential for performance)

## Threshold Tuning Guide

```python
# Test different thresholds to find optimal setting
thresholds = [0.5, 0.6, 0.7, 0.8]
for threshold in thresholds:
    config = NodeSearchConfig(relevance_threshold=threshold)
    matches = await searcher.find_related_nodes(query, config)
    print(f"Threshold {threshold}: {len(matches)} results")
```

**Guidelines:**
- **0.5**: Maximum recall, may include tangential content
- **0.6**: Balanced (default recommendation)
- **0.7**: Higher precision, may miss related content
- **0.8+**: Very high precision, only direct matches

## Example Output

```
Search Results for "revenue growth":

1. [0007] ITEM 7 - Financial Performance (Score: 0.92)
   Reason: Directly discusses revenue growth metrics and trends
   Excerpt: "Revenue increased 15% year-over-year to $385.7B..."

2. [0015] Note 1 - Revenue Recognition (Score: 0.78)
   Reason: Details revenue accounting policies affecting growth
   Excerpt: "Revenue is recognized when control transfers..."

3. [0003] Business Overview (Score: 0.71)
   Reason: Mentions growth strategy and revenue targets
   Excerpt: "Targeting 20% annual revenue growth through..."
```

## Integration with Other Components

NodeSearcher is used internally by:
- **AgenticQA**: Finds relevant sections for question answering
- **ProvenanceExtractor**: Scores all nodes for exhaustive extraction

Can be used standalone for:
- Custom retrieval workflows
- Building search interfaces
- Analyzing document coverage

## Advanced Patterns

### Pattern 1: Multi-Query Relevance
```python
# Find nodes relevant to ANY of multiple queries
queries = ["revenue", "profit", "earnings"]
all_matches = {}
for query in queries:
    matches = await searcher.find_related_nodes(query)
    for match in matches:
        all_matches[match.node.node_id] = match
```

### Pattern 2: Progressive Refinement
```python
# Start broad, then narrow
matches = await searcher.find_related_nodes("risks")
risk_node_ids = [m.node.node_id for m in matches]

# Refine to specific risk type
specific_matches = await searcher.find_related_nodes("cybersecurity")
filtered = [m for m in specific_matches if m.node.node_id in risk_node_ids]
```

## Example Code

See [examples/searcher_showcase.py](examples/searcher_showcase.py) for a complete, runnable example demonstrating:
- High precision and comprehensive search strategies
- Threshold tuning and batch processing
- Multi-query relevance and progressive refinement
- Cache optimization and performance testing

## Reference
See `examples/searcher_showcase.py` for comprehensive demonstrations.
