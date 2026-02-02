"""
Streaming example: Progress tracking and streaming responses.

This example demonstrates:
1. Indexing with progress callbacks
2. Streaming QA responses
3. Streaming provenance results
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from documentindex import (
    DocumentIndexer,
    AgenticQA,
    ProvenanceExtractor,
    IndexerConfig,
    LLMConfig,
    ProgressUpdate,
    create_azure_client,
)


SAMPLE_DOC = """
QUARTERLY REPORT

Q4 2024 Financial Results

Revenue Performance

Total revenue for Q4 2024 reached $2.5 billion, representing a 12% increase 
year-over-year. This growth was driven by strong demand across all segments.

Product revenue grew 15% to $1.8 billion.
Service revenue grew 8% to $700 million.

Operating Metrics

Gross margin improved to 65% from 62% in the prior year quarter.
Operating expenses were $800 million, representing 32% of revenue.
Operating income was $825 million with a margin of 33%.

Cash Flow and Liquidity

Operating cash flow was $600 million.
Free cash flow was $450 million after capital expenditures of $150 million.
Cash and investments totaled $3.2 billion at quarter end.

Risk Factors Update

Market conditions remain challenging with increased competition.
Supply chain constraints continue to impact certain product lines.
Regulatory requirements in key markets are evolving.
Cybersecurity threats require ongoing investment in protective measures.

Outlook

For Q1 2025, we expect revenue of $2.4-2.6 billion.
Full year 2025 guidance: revenue of $10.5-11.0 billion.
We remain committed to innovation and operational excellence.
"""


def create_progress_bar(pct: float, width: int = 40) -> str:
    """Create a text progress bar"""
    filled = int(width * pct / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:.1f}%"


async def indexing_with_progress(llm_client):
    """Example: Indexing with progress tracking"""
    print("=" * 60)
    print("EXAMPLE 1: Indexing with Progress")
    print("=" * 60)
    
    def on_progress(update: ProgressUpdate):
        """Progress callback"""
        bar = create_progress_bar(update.progress_pct)
        print(f"\r{bar} - {update.step_name}", end="", flush=True)
        if update.is_complete:
            print()  # Newline at end
    
    config = IndexerConfig(
        llm_config=llm_client.config,
        generate_summaries=True,
    )
    indexer = DocumentIndexer(config)
    
    print("\nIndexing document with progress tracking...\n")
    
    doc_index = await indexer.index_with_progress(
        text=SAMPLE_DOC,
        doc_name="Q4_Report",
        progress_callback=on_progress,
    )
    
    print(f"\nIndexing complete!")
    print(f"  Nodes created: {doc_index.get_node_count()}")
    print(f"  Chunks: {len(doc_index.chunks)}")
    
    return doc_index


async def streaming_qa(doc_index, llm_client):
    """Example: Streaming QA response"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Streaming QA Response")
    print("=" * 60)
    
    qa = AgenticQA(doc_index, llm_client=llm_client)
    
    question = "What was the revenue performance in Q4 2024?"
    print(f"\nQuestion: {question}")
    print("\nStreaming answer: ", end="", flush=True)
    
    result = await qa.answer_stream(question)
    
    # Stream the answer
    async for chunk in result.answer_stream:
        print(chunk.content, end="", flush=True)
        if chunk.is_complete:
            print("\n")  # Newlines at end
    
    print(f"Nodes visited: {len(result.nodes_visited)}")


async def streaming_provenance(doc_index, llm_client):
    """Example: Streaming provenance extraction"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Streaming Provenance Results")
    print("=" * 60)
    
    extractor = ProvenanceExtractor(doc_index, llm_client=llm_client)
    
    topic = "financial metrics and performance"
    print(f"\nExtracting evidence for: '{topic}'")
    print("\nResults as they arrive:\n")
    
    result = await extractor.extract_stream(topic)
    
    count = 0
    async for match in result.evidence_stream:
        count += 1
        print(f"[{count}] Found: {match.node.title}")
        print(f"    Score: {match.relevance_score:.2f}")
        print(f"    Reason: {match.match_reason[:80]}...")
        if match.matched_excerpts:
            print(f"    Excerpt: {match.matched_excerpts[0][:100]}...")
        print()
    
    print(f"Total matches: {count}")
    print(f"Nodes scanned: {result.total_nodes_scanned}")


async def provenance_with_progress(doc_index, llm_client):
    """Example: Provenance with progress callbacks"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Provenance with Progress")
    print("=" * 60)
    
    def on_progress(update: ProgressUpdate):
        bar = create_progress_bar(update.progress_pct)
        print(f"\r{bar} - {update.step_name}: {update.message}", end="", flush=True)
        if update.is_complete:
            print()
    
    extractor = ProvenanceExtractor(doc_index, llm_client=llm_client)
    
    topic = "risks and challenges"
    print(f"\nExtracting evidence for: '{topic}'\n")
    
    result = await extractor.extract_with_progress(
        topic,
        progress_callback=on_progress,
    )
    
    print(f"\nExtraction complete!")
    print(f"  Evidence found: {len(result.evidence)} nodes")
    print(f"  Nodes scanned: {result.total_nodes_scanned}")
    
    if result.summary:
        print(f"\nSummary:\n{result.summary[:300]}...")


async def main():
    """Run streaming examples"""
    print("\n" + "#" * 60)
    print("DocumentIndex Streaming Examples")
    print("#" * 60)
    
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
    
    # Create Azure OpenAI client once - shared across all examples
    llm_client = create_azure_client(
        deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4.1'),
        api_base=os.getenv('AZURE_OPENAI_API_BASE'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        temperature=0.0,
    )
    
    # Index with progress
    doc_index = await indexing_with_progress(llm_client)
    
    # Streaming QA
    await streaming_qa(doc_index, llm_client)
    
    # Streaming provenance
    await streaming_provenance(doc_index, llm_client)
    
    # Provenance with progress
    await provenance_with_progress(doc_index, llm_client)
    
    print("\n" + "=" * 60)
    print("Streaming examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
