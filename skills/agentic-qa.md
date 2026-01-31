# Skill: Agentic Question Answering with AgenticQA

## Overview
Intelligent, iterative question answering that navigates document structure, follows cross-references, and builds confidence-scored answers with full reasoning traces.

## Core Capability
Answer questions by reasoning through the document tree using multi-step retrieval: plan → search → read → follow references → synthesize → answer. Stops when sufficient confidence is reached.

## API Pattern

```python
from documentindex import AgenticQA, AgenticQAConfig

# Basic usage
qa = AgenticQA(doc_index, llm_client=llm_client)
result = await qa.answer("What was the revenue in 2024?")

print(result.answer)           # The answer text
print(result.confidence)       # Float 0.0-1.0
print(result.citations)        # List of Citation objects
print(result.reasoning_trace)  # List of reasoning steps

# With configuration
config = AgenticQAConfig(
    max_iterations=5,           # Max retrieval steps
    max_context_tokens=8000,    # Max context size
    follow_cross_refs=True,     # Follow "See Note X"
    generate_citations=True,    # Include citations
    confidence_threshold=0.7    # Stop when reached
)
result = await qa.answer(question, config)

# Streaming responses
result = await qa.answer_stream(question)
async for chunk in result.answer_stream:
    if chunk.content:
        print(chunk.content, end="", flush=True)
    if chunk.is_complete:
        break
```

## Key Features

### 1. Iterative Reasoning
- **Plan**: Analyzes question and plans search strategy
- **Search**: Finds most relevant sections
- **Read**: Extracts information from sections
- **Follow**: Follows cross-references when needed
- **Synthesize**: Combines findings
- **Answer**: Generates final answer when confident

### 2. Confidence Scoring
- 0.0-1.0 score indicating answer quality
- Based on information completeness and source reliability
- Configurable stopping threshold

### 3. Full Reasoning Traces
- Every step logged with action type and reasoning
- Nodes visited and findings collected
- Useful for debugging and validation

### 4. Citation Generation
- Automatic citations to source sections
- Includes excerpts and node references
- Links answer back to original text

### 5. Multi-Hop Reasoning
- Answers complex questions requiring multiple sources
- Follows references between sections
- Combines information from different parts

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_iterations` | int | 5 | Maximum reasoning steps |
| `max_context_tokens` | int | 8000 | Maximum context size |
| `follow_cross_refs` | bool | True | Follow cross-references |
| `generate_citations` | bool | True | Include citations |
| `confidence_threshold` | float | 0.7 | Stop when reached |

## Output Structure

```python
# QAResult contains:
result.answer              # String answer
result.confidence          # Float 0.0-1.0
result.citations           # List of Citation objects
result.reasoning_trace     # List of reasoning step dicts
result.nodes_visited       # List of node IDs

# Citation structure:
citation.node_id          # Source node ID
citation.node_title       # Source section title
citation.excerpt          # Relevant text excerpt
citation.relevance_score  # Score 0.0-1.0

# Reasoning step structure (dict):
step['action']            # "plan", "read_section", "follow_reference", "answer"
step['reasoning']         # Explanation of this step
step['node_id']           # Node being processed (if applicable)
step['findings']          # Information extracted (if applicable)
```

## Reasoning Actions

| Action | Description | When Used |
|--------|-------------|-----------|
| `plan` | Analyze question and plan search | First step |
| `read_section` | Extract info from a section | When relevant node found |
| `follow_reference` | Follow cross-reference | When reference needed |
| `synthesize` | Combine findings | Intermediate synthesis |
| `answer` | Generate final answer | Sufficient confidence |
| `give_up` | Admit inability to answer | Max iterations reached |

## Common Use Cases

### Use Case 1: Quick Questions
```python
# Fast answers with lower confidence threshold
config = AgenticQAConfig(
    max_iterations=3,
    confidence_threshold=0.6
)
result = await qa.answer("What was Q4 revenue?", config)
```

### Use Case 2: Thorough Analysis
```python
# Deep analysis with high confidence requirement
config = AgenticQAConfig(
    max_iterations=10,
    confidence_threshold=0.85,
    max_context_tokens=12000
)
result = await qa.answer("Analyze competitive positioning", config)
```

### Use Case 3: Multi-Hop Questions
```python
# Questions requiring multiple sources
config = AgenticQAConfig(
    max_iterations=8,
    follow_cross_refs=True
)
result = await qa.answer(
    "What percentage of revenue comes from the fastest growing segment?",
    config
)
```

## Best Practices

1. **Use default settings initially**: Adjust based on question complexity
2. **Check confidence scores**: Low confidence (<0.6) = uncertain answer
3. **Review reasoning traces**: Understand how answer was derived
4. **Enable cross-reference following**: Critical for comprehensive answers
5. **Adjust iterations for question complexity**: Simple = 3-5, Complex = 8-10

## Performance Considerations

- **Answer time**: ~5-15 seconds depending on complexity
- **API costs**: ~$0.02-0.10 per question depending on iterations
- **Recommended settings**:
  - Simple questions: `max_iterations=3`, `confidence_threshold=0.6`
  - Standard questions: `max_iterations=5`, `confidence_threshold=0.7`
  - Complex questions: `max_iterations=10`, `confidence_threshold=0.8`

## Confidence Score Interpretation

| Score | Meaning | Action |
|-------|---------|--------|
| 0.9-1.0 | High confidence | Trust the answer |
| 0.8-0.9 | Good confidence | Generally reliable |
| 0.7-0.8 | Moderate confidence | Verify if critical |
| 0.6-0.7 | Low confidence | Use with caution |
| < 0.6 | Very low confidence | Likely incomplete/uncertain |

## Example Output

```
Question: What are the main revenue drivers?

Answer (Confidence: 0.88):
The main revenue drivers are:
1. Cloud services (65% of revenue, growing 25% YoY)
2. Enterprise solutions (25% of revenue, stable growth)
3. Consumer products (10% of revenue, declining)

Citations:
- [0007] ITEM 7 - Management Discussion: "Cloud services represent 65%..."
- [0015] Note 15 - Segment Information: "Cloud segment grew 25%..."

Reasoning Trace:
Step 1 [plan]: Need to find revenue segments and growth factors
Step 2 [read_section]: Reading ITEM 7 - Management Discussion
  Findings: Found three segments with percentages
Step 3 [follow_reference]: Following reference to Note 15
  Findings: Detailed growth rates for each segment
Step 4 [synthesize]: Combining segment data and growth rates
Step 5 [answer]: Generated final answer with 88% confidence
```

## Advanced Patterns

### Pattern 1: Batch Questions
```python
questions = [
    "What was revenue?",
    "What are the risks?",
    "Who are the competitors?"
]

for question in questions:
    result = await qa.answer(question)
    print(f"Q: {question}")
    print(f"A: {result.answer} (confidence: {result.confidence:.2f})\n")
```

### Pattern 2: Interactive Q&A
```python
context = []
for user_question in user_questions:
    result = await qa.answer(user_question)

    if result.confidence < 0.7:
        print(f"⚠️ Low confidence: {result.confidence:.2f}")
        print("Consider rephrasing or asking more specific question")

    context.append((user_question, result.answer))
```

### Pattern 3: Progressive Refinement
```python
# Start broad
result1 = await qa.answer("What are the risks?")

# Drill down on specific risk
result2 = await qa.answer("Tell me more about cybersecurity risks")

# Compare findings
print(f"Broad search: {len(result1.citations)} sources")
print(f"Specific search: {len(result2.citations)} sources")
```

## Integration with Other Components

AgenticQA uses:
- **NodeSearcher**: For finding relevant sections
- **CrossReferenceFollower**: For following "See Note X" references
- **DocumentIndex**: For accessing node text and structure

Can be combined with:
- **ProvenanceExtractor**: QA for direct questions, Provenance for exhaustive evidence
- **Custom workflows**: Use reasoning traces to build specialized retrievers

## Debugging Tips

```python
# Enable full reasoning trace display
result = await qa.answer(question)

for i, step in enumerate(result.reasoning_trace, 1):
    print(f"Step {i} [{step['action']}]")
    print(f"  Reasoning: {step['reasoning']}")
    if step.get('findings'):
        print(f"  Findings: {step['findings']}")
```

## Reference
See `examples/agentic_qa_tutorial.py` for comprehensive demonstrations.
