"""
Basic usage example: Indexing and querying a document.

This example demonstrates:
1. Loading and indexing a document
2. Searching for specific topics
3. Asking questions with agentic QA
4. Extracting all evidence about a topic
"""

import asyncio
from documentindex import (
    DocumentIndexer,
    NodeSearcher,
    AgenticQA,
    ProvenanceExtractor,
    IndexerConfig,
    LLMConfig,
)

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


async def basic_indexing_example():
    """Example: Basic document indexing"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Document Indexing")
    print("=" * 60)
    
    # Create indexer with configuration
    config = IndexerConfig(
        llm_config=LLMConfig(model="gpt-4o", temperature=0.0),
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
    for node in doc_index.structure:
        print(f"  [{node.node_id}] {node.title}")
        for child in node.children:
            print(f"    [{child.node_id}] {child.title}")
    
    return doc_index


async def search_example(doc_index):
    """Example: Searching for related nodes"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Node Search")
    print("=" * 60)
    
    searcher = NodeSearcher(doc_index)
    
    # Search for nodes related to a topic
    query = "cybersecurity risks and data protection"
    print(f"\nSearching for: '{query}'")
    
    matches = await searcher.find_related_nodes(query)
    
    print(f"\nFound {len(matches)} matching nodes:")
    for match in matches[:5]:
        print(f"\n  [{match.node.node_id}] {match.node.title}")
        print(f"    Relevance: {match.relevance_score:.2f}")
        print(f"    Reason: {match.match_reason[:100]}...")


async def qa_example(doc_index):
    """Example: Question Answering"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Agentic Question Answering")
    print("=" * 60)
    
    qa = AgenticQA(doc_index)
    
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


async def provenance_example(doc_index):
    """Example: Provenance Extraction"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Provenance Extraction")
    print("=" * 60)
    
    extractor = ProvenanceExtractor(doc_index)
    
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


async def multi_topic_example(doc_index):
    """Example: Multiple topic extraction"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Multi-Topic Extraction")
    print("=" * 60)
    
    extractor = ProvenanceExtractor(doc_index)
    
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
    
    # Run examples
    doc_index = await basic_indexing_example()
    await search_example(doc_index)
    await qa_example(doc_index)
    await provenance_example(doc_index)
    await multi_topic_example(doc_index)
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
