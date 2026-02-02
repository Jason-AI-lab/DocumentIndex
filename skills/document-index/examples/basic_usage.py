"""
Basic usage example: Indexing and querying a document.

This example demonstrates:
1. Loading and indexing a document
2. Searching for specific topics
3. Asking questions with agentic QA
4. Extracting all evidence about a topic
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv
from documentindex import (
    DocumentIndexer,
    NodeSearcher,
    AgenticQA,
    ProvenanceExtractor,
    IndexerConfig,
    create_azure_client,
    create_bedrock_client,
)
from typing import Optional

# Sample 10-K document (abbreviated for example)
SAMPLE_10K = """
FORM 10-K

PART I

ITEM 1. BUSINESS

Overview

TechCorp Inc. ("TechCorp," "the Company," "we," or "us") is a leading provider of 
enterprise software solutions. Founded in 2005, we have grown to serve over 5,000 
enterprise customers worldwide.

Our Revenue for fiscal year 2024 was $8.5 billion, representing a 15% increase 
from the prior year. Net Income was $1.2 billion.

Products and Services

We offer cloud-based enterprise solutions:
1. TechCloud Platform - Our flagship product
2. DataSync - Enterprise data integration
3. SecureAuth - Identity management

For more details, see Note 15 in the financial statements.

ITEM 1A. RISK FACTORS

Market Competition

The enterprise software market is highly competitive. We face competition from 
established vendors and emerging startups. Failure to compete effectively could 
materially harm our business.

Cybersecurity Risks

Our business relies on secure data transmission and storage. A significant 
security breach could damage our reputation and financial results. We invest 
heavily in security measures but cannot guarantee prevention of all breaches.

Climate and Environmental Risks

Climate change may affect our operations. Our data centers consume significant 
energy, and we are committed to reducing our environmental footprint. Extreme 
weather events could disrupt our services.

Regulatory Compliance

We are subject to data privacy regulations including GDPR and CCPA. Changes in 
regulatory requirements could increase compliance costs.

PART II

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Financial Highlights

Fiscal 2024 was a year of strong performance:
- Revenue: $8.5 billion (+15% YoY)
- Gross Margin: 72% (up from 70%)
- Operating Income: $1.8 billion
- Free Cash Flow: $2.1 billion

We continue to invest in R&D to maintain our competitive position.

Segment Performance

TechCloud Platform generated $5.2 billion in revenue, up 18%.
DataSync contributed $2.1 billion, up 12%.
SecureAuth generated $1.2 billion, up 8%.

NOTE 15. SEGMENT INFORMATION

The Company operates in three reportable segments:
1. TechCloud Platform
2. DataSync  
3. SecureAuth

Each segment is managed separately based on the products and services offered.
"""

DOCUMENTINDEX_LLM_PROVIDER = "bedrock" # "azure" or "bedrock"

def create_llm_client_from_env():
    """Build an LLM client based on environment configuration."""
    provider = os.getenv("DOCUMENTINDEX_LLM_PROVIDER", DOCUMENTINDEX_LLM_PROVIDER).lower()
    print(f"Using LLM provider: {provider}")
    temperature_override = os.getenv("DOCUMENTINDEX_LLM_TEMPERATURE")
    if temperature_override:
        try:
            temperature = float(temperature_override)
        except ValueError as exc:
            raise ValueError("DOCUMENTINDEX_LLM_TEMPERATURE must be a numeric value") from exc
    else:
        temperature = 0.0
    if provider == "bedrock":
        model = os.getenv("AWS_BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
        print(f"Using model: {model}")
        region = os.getenv("AWS_REGION", "us-east-1")
        return create_bedrock_client(model=model, region=region, temperature=temperature)
    if provider == "azure":
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
        print(f"Using deployment: {deployment}")
        if not temperature_override and deployment.lower().startswith("gpt-5"):
            temperature = 1.0  # Azure GPT-5 deployments require temperature=1
        return create_azure_client(
            deployment_name=deployment,
            api_base=os.getenv("AZURE_OPENAI_API_BASE"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=temperature,
        )
    raise ValueError(
        f"Unsupported DOCUMENTINDEX_LLM_PROVIDER '{provider}'. Expected 'bedrock' or 'azure'."
    )


async def basic_indexing_example(llm_client, max_depth: Optional[int] = None):
    """Example: Basic document indexing

    Args:
        llm_client: The LLM client to use
        max_depth: Maximum depth to display in tree structure (None = show all)
    """
    print("=" * 60)
    print("EXAMPLE 1: Basic Document Indexing")
    print("=" * 60)

    def print_tree_structure(nodes: list, max_depth: Optional[int] = None, current_depth: int = 0, indent: str = ""):
        """Print tree structure with configurable depth.

        Args:
            nodes: List of nodes to print
            max_depth: Maximum depth to display (None = show all)
            current_depth: Current recursion depth
            indent: Current indentation string
        """
        if not nodes:
            if current_depth == 0:
                print("  (No structure found)")
            return

        # Handle invalid depth values
        if max_depth is not None and max_depth <= 0:
            if current_depth == 0:
                print("  (Depth set to 0, no nodes shown)")
            return

        for node in nodes:
            # Handle very long titles
            title = node.title
            if len(title) > 80:
                title = title[:77] + "..."

            print(f"{indent}[{node.node_id}] {title}")

            # Continue deeper if allowed
            if max_depth is None or current_depth < max_depth - 1:
                if node.children:
                    print_tree_structure(
                        node.children,
                        max_depth,
                        current_depth + 1,
                        indent + "  "
                    )
            elif node.children and max_depth and current_depth == max_depth - 1:
                # Show count of hidden children
                child_count = len(node.children)
                if child_count > 0:
                    plural = "node" if child_count == 1 else "nodes"
                    print(f"{indent}  ... ({child_count} child {plural} not shown)")

    # Create indexer with Azure OpenAI configuration
    config = IndexerConfig(
        llm_config=llm_client.config,
        generate_summaries=True,
    )
    indexer = DocumentIndexer(config)
    
    # Index the document
    print("\nIndexing document...")
    doc_index = await indexer.index(
        text=SAMPLE_10K,
        doc_name="TechCorp_10K_2024",
    )
    
    # Print structure
    print(f"\nDocument: {doc_index.doc_name}")
    print(f"Type: {doc_index.doc_type.value}")
    print(f"Total nodes: {doc_index.get_node_count()}")
    print(f"Total chunks: {len(doc_index.chunks)}")
    
    print("\nDocument Structure:")
    print_tree_structure(doc_index.structure, max_depth)

    # Add a note about the depth parameter when limited
    if max_depth:
        print(f"\n(Showing up to depth {max_depth}. Total nodes in document: {doc_index.get_node_count()})")

    # print("\nDocument index")
    # print(doc_index)
    
    return doc_index


async def search_example(doc_index, llm_client):
    """Example: Searching for related nodes"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Node Search")
    print("=" * 60)
    
    searcher = NodeSearcher(doc_index, llm_client=llm_client)
    
    # Search for nodes related to a topic
    query = "cybersecurity risks and data protection"
    print(f"\nSearching for: '{query}'")
    
    matches = await searcher.find_related_nodes(query)
    
    print(f"\nFound {len(matches)} matching nodes:")
    for match in matches[:5]:
        print(f"\n  [{match.node.node_id}] {match.node.title}")
        print(f"    Relevance: {match.relevance_score:.2f}")
        print(f"    Reason: {match.match_reason[:100]}...")


async def qa_example(doc_index, llm_client):
    """Example: Question Answering"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Agentic Question Answering")
    print("=" * 60)
    
    qa = AgenticQA(doc_index, llm_client=llm_client)
    
    # Ask a question
    question = "What was TechCorp's revenue in 2024 and how did it compare to the previous year?"
    print(f"\nQuestion: {question}")
    
    result = await qa.answer(question)
    
    print(f"\nAnswer: {result.answer}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Nodes visited: {len(result.nodes_visited)}")
    
    if result.citations:
        print("\nCitations:")
        for citation in result.citations[:3]:
            print(f"  - {citation.node_title}: {citation.excerpt[:100]}...")


async def provenance_example(doc_index, llm_client):
    """Example: Provenance Extraction"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Provenance Extraction")
    print("=" * 60)
    
    extractor = ProvenanceExtractor(doc_index, llm_client=llm_client)
    
    # Extract all evidence about a topic
    topic = "climate change and environmental sustainability"
    print(f"\nExtracting evidence for: '{topic}'")
    
    result = await extractor.extract_all(topic)
    
    print(f"\nNodes scanned: {result.total_nodes_scanned}")
    print(f"Evidence found: {len(result.evidence)} nodes")
    
    if result.evidence:
        print("\nRelevant sections:")
        for match in result.evidence[:5]:
            print(f"\n  [{match.node.node_id}] {match.node.title}")
            print(f"    Relevance: {match.relevance_score:.2f}")
            if match.matched_excerpts:
                print(f"    Excerpt: {match.matched_excerpts[0][:150]}...")
    
    if result.summary:
        print(f"\nSummary: {result.summary[:300]}...")


async def multi_topic_example(doc_index, llm_client):
    """Example: Multiple topic extraction"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Multi-Topic Extraction")
    print("=" * 60)
    
    extractor = ProvenanceExtractor(doc_index, llm_client=llm_client)
    
    # Define multiple topics
    topics = {
        "financial_performance": "revenue, profit, earnings, financial results",
        "risk_factors": "risks, threats, challenges, vulnerabilities",
        "products_services": "products, services, offerings, solutions",
    }
    
    print(f"\nExtracting evidence for {len(topics)} topics...")
    
    results = await extractor.extract_by_category(topics)
    
    print("\nResults by topic:")
    for topic_name, result in results.items():
        print(f"\n  {topic_name}:")
        print(f"    Evidence nodes: {len(result.evidence)}")
        if result.evidence:
            top_match = result.evidence[0]
            print(f"    Top match: {top_match.node.title} (score: {top_match.relevance_score:.2f})")


async def main():
    """Run all examples"""
    print("\n" + "#" * 60)
    print("DocumentIndex Usage Examples")
    print("#" * 60)
    
    # Load environment variables from .env file
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
    
    llm_client = create_llm_client_from_env()

    # Optional: Get depth from environment
    display_depth = None  # Default: show all levels
    depth_env = os.getenv("DOCUMENTINDEX_DISPLAY_DEPTH")
    if depth_env:
        try:
            display_depth = int(depth_env)
        except ValueError:
            print(f"Warning: Invalid DOCUMENTINDEX_DISPLAY_DEPTH '{depth_env}', using default")

    # Run examples
    doc_index = await basic_indexing_example(llm_client, max_depth=display_depth)
    await search_example(doc_index, llm_client)
    await qa_example(doc_index, llm_client)
    await provenance_example(doc_index, llm_client)
    await multi_topic_example(doc_index, llm_client)
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
