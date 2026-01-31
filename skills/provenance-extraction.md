# Skill: Provenance Extraction with ProvenanceExtractor

## Overview
Exhaustive evidence extraction that scans 100% of document nodes to find ALL relevant content for topics. Unlike AgenticQA (which stops when confident), this ensures complete coverage for compliance, audits, and research.

## Core Capability
Systematically evaluate every node in a document tree against one or more topics, collecting all evidence with relevance scores, excerpts, and optional summaries. Provides progress tracking for long operations.

## API Pattern

```python
from documentindex import ProvenanceExtractor, ProvenanceConfig

# Basic usage
extractor = ProvenanceExtractor(doc_index, llm_client=llm_client)
result = await extractor.extract_all("climate change risks")

print(f"Found {len(result.evidence)} relevant sections")
print(f"Scanned {result.total_nodes_scanned} nodes")
print(f"Coverage: {result.scan_coverage:.0%}")

# With configuration
config = ProvenanceConfig(
    relevance_threshold=0.6,        # Min score to include
    extract_excerpts=True,          # Extract relevant excerpts
    max_excerpts_per_node=3,        # Max excerpts per node
    generate_summary=True,          # Generate summary of findings
    parallel_workers=3,             # Concurrent LLM calls
    batch_size=10                   # Nodes per batch
)
result = await extractor.extract_all("ESG initiatives", config)

# Multi-category extraction
categories = {
    "climate": "climate change and environmental risks",
    "cyber": "cybersecurity and data protection",
    "regulatory": "regulatory compliance requirements"
}
results = await extractor.extract_by_category(categories)

# Progress tracking
def on_progress(update):
    print(f"[{update.progress_pct:.0f}%] {update.step_name}")

result = await extractor.extract_with_progress(
    topic="climate risks",
    progress_callback=on_progress
)

# Streaming results
async for match in extractor.extract_stream("financial risks"):
    print(f"Found: [{match.node.node_id}] {match.node.title}")
```

## Key Features

### 1. Exhaustive Scanning
- Evaluates 100% of nodes in document tree
- Guarantees no relevant content is missed
- Reports scan coverage statistics

### 2. Multi-Category Analysis
- Extract evidence for multiple topics concurrently
- Identifies overlap between categories
- Efficient parallel processing

### 3. Evidence Scoring
- Each match scored 0.0-1.0 for relevance
- Configurable threshold for inclusion
- Helps prioritize most relevant sections

### 4. Excerpt Extraction
- Extracts relevant text passages from each node
- Configurable number of excerpts per node
- Useful for quick review without full text

### 5. Summary Generation
- Optional LLM-generated summary of all findings
- Synthesizes evidence across all matches
- Useful for executive reports

### 6. Progress Tracking
- Real-time updates during long operations
- Shows nodes processed and time remaining
- Essential for large documents

### 7. Streaming Results
- Yields results as they're found
- Enables real-time processing
- Reduces memory usage for large result sets

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `relevance_threshold` | float | 0.6 | Minimum relevance score |
| `extract_excerpts` | bool | True | Extract text excerpts |
| `max_excerpts_per_node` | int | 3 | Max excerpts per match |
| `generate_summary` | bool | False | Generate findings summary |
| `parallel_workers` | int | 3 | Concurrent LLM calls |
| `batch_size` | int | 10 | Nodes to process per batch |

## Output Structure

```python
# ProvenanceResult contains:
result.evidence              # List of NodeMatch objects
result.total_nodes_scanned   # Int - nodes evaluated
result.scan_coverage         # Float 0.0-1.0 (usually 1.0)
result.summary               # String (if generate_summary=True)

# NodeMatch structure:
match.node                   # TreeNode object
match.relevance_score        # Float 0.0-1.0
match.match_reason           # String explanation
match.excerpts               # List of relevant text excerpts

# For multi-category extraction:
results = {
    "climate": ProvenanceResult(...),
    "cyber": ProvenanceResult(...),
    "regulatory": ProvenanceResult(...)
}
```

## Common Use Cases

### Use Case 1: Compliance Audit
```python
# Find ALL mentions of regulatory requirements
config = ProvenanceConfig(
    relevance_threshold=0.5,  # Cast wide net
    extract_excerpts=True,
    generate_summary=True
)
result = await extractor.extract_all(
    "regulatory compliance requirements",
    config
)

# Export for review
for match in result.evidence:
    print(f"[{match.node.node_id}] {match.node.title}")
    for excerpt in match.excerpts:
        print(f"  - {excerpt}")
```

### Use Case 2: Risk Analysis
```python
# Analyze different risk categories
risk_categories = {
    "climate": "climate change and environmental risks",
    "cyber": "cybersecurity threats and data breaches",
    "supply_chain": "supply chain disruptions",
    "regulatory": "regulatory changes and compliance",
    "financial": "financial risks and market volatility"
}

results = await extractor.extract_by_category(risk_categories)

# Compare coverage
for category, result in results.items():
    print(f"{category}: {len(result.evidence)} relevant sections")
```

### Use Case 3: Topic Research
```python
# Deep dive into specific topic with high precision
config = ProvenanceConfig(
    relevance_threshold=0.7,  # High precision
    extract_excerpts=True,
    max_excerpts_per_node=5,  # More excerpts
    generate_summary=True
)

result = await extractor.extract_all("AI and machine learning initiatives", config)
print(result.summary)  # Executive summary of findings
```

### Use Case 4: Progress Monitoring
```python
# Track progress for large document
def progress_callback(update):
    pct = update.progress_pct
    print(f"[{'█' * int(pct/10)}{'░' * (10-int(pct/10))}] {pct:.0f}%")

result = await extractor.extract_with_progress(
    topic="ESG initiatives",
    progress_callback=progress_callback
)
```

## Best Practices

1. **Use appropriate threshold**: 0.5-0.6 for broad coverage, 0.7+ for precision
2. **Enable excerpts**: Makes review much faster than reading full text
3. **Use multi-category for related topics**: More efficient than separate calls
4. **Generate summaries for reports**: Saves manual synthesis time
5. **Track progress for large docs**: Provides visibility into long operations

## Performance Considerations

- **Extraction time**: ~30-90 seconds for typical 30-40 node document
- **API costs**: ~$0.05-0.20 per extraction depending on document size
- **Memory usage**: Minimal with streaming, moderate with full results
- **Recommended settings**:
  - Small docs (<20 nodes): `parallel_workers=5`, `batch_size=15`
  - Medium docs (20-50 nodes): `parallel_workers=3`, `batch_size=10` (default)
  - Large docs (>50 nodes): `parallel_workers=2`, `batch_size=8`, use streaming

## Threshold Tuning Guide

```python
# Test different thresholds
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

for threshold in thresholds:
    config = ProvenanceConfig(relevance_threshold=threshold)
    result = await extractor.extract_all(topic, config)

    print(f"\nThreshold {threshold}:")
    print(f"  Matches: {len(result.evidence)}")
    print(f"  Avg score: {sum(m.relevance_score for m in result.evidence)/len(result.evidence):.2f}")
```

**Guidelines:**
- **0.5**: Maximum coverage, may include tangential mentions
- **0.6**: Balanced (default), good for most use cases
- **0.7**: Higher precision, focused results
- **0.8+**: Very high precision, only direct/detailed coverage

## Export Patterns

### Export to JSON
```python
import json

# Export for programmatic processing
with open("evidence.json", "w") as f:
    json.dump({
        "topic": "climate risks",
        "total_matches": len(result.evidence),
        "evidence": [
            {
                "node_id": m.node.node_id,
                "title": m.node.title,
                "score": m.relevance_score,
                "excerpts": m.excerpts
            }
            for m in result.evidence
        ]
    }, f, indent=2)
```

### Export to CSV
```python
import csv

# Export for spreadsheet analysis
with open("evidence.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Node ID", "Title", "Score", "Excerpt Count"])

    for match in result.evidence:
        writer.writerow([
            match.node.node_id,
            match.node.title,
            f"{match.relevance_score:.2f}",
            len(match.excerpts)
        ])
```

### Export to Markdown Report
```python
# Generate human-readable report
with open("report.md", "w") as f:
    f.write(f"# Evidence Report: {topic}\n\n")
    f.write(f"**Summary**: {result.summary}\n\n")
    f.write(f"**Total Matches**: {len(result.evidence)}\n\n")

    for i, match in enumerate(result.evidence, 1):
        f.write(f"## {i}. [{match.node.node_id}] {match.node.title}\n\n")
        f.write(f"**Relevance**: {match.relevance_score:.2f}\n\n")
        f.write(f"**Reason**: {match.match_reason}\n\n")

        if match.excerpts:
            f.write("**Key Excerpts**:\n\n")
            for excerpt in match.excerpts:
                f.write(f"- {excerpt}\n")
            f.write("\n")
```

## Example Output

```
Extracting evidence for "ESG initiatives":

Progress: [████████░░] 80% (32/40 nodes)

Evidence Found (8 matches):

1. [0009] Sustainability Report (Score: 0.95) 🔴
   "Reduced carbon emissions by 30% through renewable energy..."
   "Achieved zero waste to landfill at 15 manufacturing sites..."
   "Committed $100M to sustainable supply chain initiatives..."

2. [0012] Risk Factors - Climate (Score: 0.82) 🟠
   "Climate regulations may increase operational costs..."
   "Transition to renewable energy requires significant investment..."

3. [0018] Governance Practices (Score: 0.78) 🟠
   "Board sustainability committee oversees ESG strategy..."

Summary:
Found 8 ESG-related sections covering emissions reduction (3 sections),
renewable energy (2 sections), and governance (3 sections). Company has
committed to 30% emissions reduction and $100M in sustainability investments.
```

## Integration with Other Components

ProvenanceExtractor complements:
- **AgenticQA**: Use QA for direct questions, Provenance for exhaustive evidence
- **NodeSearcher**: Uses NodeSearcher internally for scoring

Typical workflow:
1. **Index document** with DocumentIndexer
2. **Quick questions** with AgenticQA
3. **Comprehensive evidence** with ProvenanceExtractor
4. **Export findings** for reports/analysis

## When to Use Provenance vs AgenticQA

| Use Provenance When | Use AgenticQA When |
|---------------------|-------------------|
| Need ALL evidence | Need direct answer |
| Compliance/audit work | Exploratory questions |
| Research deep dive | Time-sensitive queries |
| Multi-category analysis | Single focused question |
| Building reports | Interactive Q&A |
| Ensuring completeness | Efficiency matters |

## Reference
See `examples/provenance_patterns.py` for comprehensive demonstrations including:
- Single-topic extraction
- Multi-category analysis
- Threshold tuning
- Summary generation
- Progress tracking
- Streaming extraction
- Export formats (JSON, CSV, Markdown)
