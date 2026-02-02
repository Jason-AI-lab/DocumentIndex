"""
DocumentIndexer Deep Dive - Comprehensive indexing examples

This example demonstrates:
1. Basic indexing with different document types
2. Configuration options and their effects
3. Hierarchical structure analysis with summaries
4. Metadata extraction capabilities
5. Cross-reference resolution
6. Performance optimization tips
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from documentindex import (
    DocumentIndexer,
    IndexerConfig,
    ChunkConfig,
    LLMConfig,
    TreeNode,
    DocumentIndex,
    create_azure_client,
    create_bedrock_client,
)

# Sample Documents
SAMPLE_10K = """FORM 10-K
ANNUAL REPORT PURSUANT TO SECTION 13 OR 15(d)
OF THE SECURITIES EXCHANGE ACT OF 1934

For the fiscal year ended December 31, 2024

TECHCORP INC.
(Exact name of registrant as specified in its charter)

PART I

ITEM 1. BUSINESS

Overview
TechCorp Inc. is a global leader in cloud computing and enterprise software solutions.
Founded in 2010, we have grown to serve over 50,000 customers worldwide with annual
revenue of $15.2 billion in fiscal year 2024, representing a 15% increase from the
prior year.

Our Products and Services
We offer three main product lines:
1. CloudPlatform - Our flagship infrastructure-as-a-service offering
2. EnterpriseSuite - Comprehensive business management software
3. DataAnalytics - Advanced analytics and machine learning platform

See Note 15 for detailed segment information.

ITEM 1A. RISK FACTORS

The following risk factors may materially affect our business:

Market Competition
The cloud computing market is highly competitive. Major competitors include
established technology companies with significant resources. See Item 7 for
competitive analysis.

Cybersecurity Risks
We face constant cybersecurity threats that could compromise customer data.
Refer to our cybersecurity framework in Item 1C.

Climate and Environmental Risks
Climate change regulations may increase our operating costs. See our ESG
report in Item 7A for mitigation strategies.

PART II

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Financial Performance
Revenue increased 15% year-over-year to $15.2 billion, driven primarily by:
- CloudPlatform growth of 25% to $9.9 billion
- EnterpriseSuite growth of 10% to $3.8 billion
- DataAnalytics growth of 5% to $1.5 billion

Operating income was $3.2 billion, a 20% increase, reflecting improved margins.

ITEM 7A. ESG INITIATIVES

We have reduced carbon emissions by 30% through renewable energy adoption.
Employee diversity increased with 45% of new hires from underrepresented groups.

NOTE 15. SEGMENT INFORMATION

Revenue by segment:
- CloudPlatform: 65% of total revenue
- EnterpriseSuite: 25% of total revenue
- DataAnalytics: 10% of total revenue

Geographic distribution:
- Americas: 60%
- EMEA: 25%
- APAC: 15%
"""

SAMPLE_EARNINGS_CALL = """TECHCORP Q4 2024 EARNINGS CALL TRANSCRIPT

Date: January 30, 2025
Participants: CEO Jane Smith, CFO John Doe

OPERATOR: Good afternoon and welcome to TechCorp's Q4 2024 earnings call.

CEO JANE SMITH: Thank you. I'm pleased to report another strong quarter with
revenue of $4.1 billion, up 18% year-over-year.

Our cloud business continues to outperform, growing 28% in Q4. We added
2,500 new enterprise customers, bringing our total to over 50,000.

Looking ahead to 2025, we expect continued momentum. See our detailed guidance
in the earnings release.

CFO JOHN DOE: Thanks Jane. Let me provide more detail on our financial results.

Q4 revenue breakdown:
- CloudPlatform: $2.7 billion (+28%)
- EnterpriseSuite: $1.0 billion (+12%)
- DataAnalytics: $0.4 billion (+8%)

Operating margin improved to 22%, up from 20% last year. Free cash flow
was $1.2 billion.

For Q1 2025, we guide to revenue of $4.2-4.3 billion.

Q&A SESSION

ANALYST 1: Can you discuss the competitive landscape?

CEO: The market remains competitive, but our integrated platform gives us
an advantage. Customer retention is at 95%.

ANALYST 2: What about international expansion?

CEO: APAC is our fastest growing region at 35% growth. We're investing
heavily in local data centers. Refer to our investor deck for details.
"""

SAMPLE_RESEARCH_REPORT = """EQUITY RESEARCH REPORT
TechCorp Inc. (TECH)
Rating: BUY | Price Target: $150

EXECUTIVE SUMMARY
We initiate coverage of TechCorp with a BUY rating and $150 price target,
representing 25% upside. Key investment thesis:

1. Market Leadership
TechCorp holds 22% market share in cloud infrastructure, second only to
the market leader. See industry analysis in Section 3.

2. Financial Strength
- Revenue CAGR of 18% over past 5 years
- Operating margins expanding from 18% to 22%
- Strong balance sheet with $8B cash

3. Growth Catalysts
- AI integration across product suite
- International expansion, particularly in APAC
- New product launches planned for H2 2025

RISKS
- Intense competition from tech giants
- Regulatory scrutiny increasing
- Macroeconomic headwinds

See full analysis and financial models in appendix.
"""


def print_section_header(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_tree_with_details(
    nodes: list[TreeNode],
    doc_index: DocumentIndex,
    indent: str = "",
    max_depth: Optional[int] = None,
    current_depth: int = 0,
    show_summaries: bool = True,
    show_metadata: bool = True,
    show_crossrefs: bool = True
):
    """Print tree structure with detailed information"""
    if max_depth and current_depth >= max_depth:
        return

    for node in nodes:
        # Basic node info
        print(f"{indent}[{node.node_id}] {node.title}")

        # Show text span info
        print(f"{indent}    📍 Spans: chars {node.text_span.start_char:,}-{node.text_span.end_char:,}, "
              f"chunks {node.text_span.start_chunk}-{node.text_span.end_chunk}")

        # Show summary if available
        if show_summaries and node.summary:
            summary_preview = node.summary[:100] + "..." if len(node.summary) > 100 else node.summary
            print(f"{indent}    📝 Summary: {summary_preview}")

        # Show metadata if available
        if show_metadata and node.node_metadata:
            print(f"{indent}    🏷️  Metadata: {json.dumps(node.node_metadata, indent=0)}")

        # Show cross-references
        if show_crossrefs and node.cross_references:
            refs = [f'"{ref.reference_text}"→{ref.target_node_id or "?"}' for ref in node.cross_references]
            print(f"{indent}    🔗 Cross-refs: {', '.join(refs)}")

        # Add spacing
        if node.children or (node.summary or node.node_metadata or node.cross_references):
            print()

        # Recurse for children
        if node.children:
            print_tree_with_details(
                node.children,
                doc_index,
                indent + "  ",
                max_depth,
                current_depth + 1,
                show_summaries,
                show_metadata,
                show_crossrefs
            )


def analyze_tree_structure(doc_index: DocumentIndex) -> Dict[str, Any]:
    """Analyze the tree structure and return statistics"""
    stats = {
        "total_nodes": doc_index.get_node_count(),
        "total_chunks": len(doc_index.chunks),
        "max_depth": 0,
        "nodes_by_level": {},
        "nodes_with_summaries": 0,
        "nodes_with_metadata": 0,
        "nodes_with_crossrefs": 0,
        "leaf_nodes": 0
    }

    def analyze_node(node: TreeNode, depth: int = 0):
        # Update max depth
        stats["max_depth"] = max(stats["max_depth"], depth)

        # Count by level
        if depth not in stats["nodes_by_level"]:
            stats["nodes_by_level"][depth] = 0
        stats["nodes_by_level"][depth] += 1

        # Count features
        if node.summary:
            stats["nodes_with_summaries"] += 1
        if node.node_metadata:
            stats["nodes_with_metadata"] += 1
        if node.cross_references:
            stats["nodes_with_crossrefs"] += 1
        if not node.children:
            stats["leaf_nodes"] += 1

        # Recurse
        for child in node.children:
            analyze_node(child, depth + 1)

    for root in doc_index.structure:
        analyze_node(root)

    return stats


async def example1_basic_indexing(llm_client):
    """Example 1: Basic indexing with full configuration"""
    print_section_header("EXAMPLE 1: Basic Indexing with Full Configuration")

    config = IndexerConfig(
        llm_config=llm_client.config,
        generate_summaries=True,
        extract_metadata=True,
        resolve_cross_refs=True,
        max_concurrent_summaries=3
    )

    indexer = DocumentIndexer(config)

    print("\nIndexing document with full configuration...")
    print(f"Configuration: {json.dumps(config.__dict__, default=str, indent=2)}")

    start_time = time.time()
    doc_index = await indexer.index(
        text=SAMPLE_10K,
        doc_name="TechCorp_10K_2024"
    )
    elapsed = time.time() - start_time

    print(f"\n✅ Indexing completed in {elapsed:.2f} seconds")
    print(f"Document type detected: {doc_index.doc_type.value}")

    # Show tree structure with all details
    print("\nDocument Structure with Full Details:")
    print_tree_with_details(doc_index.structure, doc_index, max_depth=3)

    # Show statistics
    stats = analyze_tree_structure(doc_index)
    print("\nTree Structure Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Nodes with summaries: {stats['nodes_with_summaries']}")
    print(f"  Nodes with metadata: {stats['nodes_with_metadata']}")
    print(f"  Nodes with cross-refs: {stats['nodes_with_crossrefs']}")
    print(f"  Leaf nodes: {stats['leaf_nodes']}")
    print(f"  Distribution by level: {stats['nodes_by_level']}")

    # Show document metadata
    if doc_index.metadata:
        print("\nDocument Metadata:")
        print(json.dumps(doc_index.metadata.to_dict(), indent=2))

    return doc_index


async def example2_configuration_comparison(llm_client):
    """Example 2: Compare different configuration options"""
    print_section_header("EXAMPLE 2: Configuration Impact Analysis")

    configurations = [
        ("Minimal", IndexerConfig(
            llm_config=llm_client.config,
            generate_summaries=False,
            extract_metadata=False,
            resolve_cross_refs=False
        )),
        ("Summaries Only", IndexerConfig(
            llm_config=llm_client.config,
            generate_summaries=True,
            extract_metadata=False,
            resolve_cross_refs=False
        )),
        ("Full Features", IndexerConfig(
            llm_config=llm_client.config,
            generate_summaries=True,
            extract_metadata=True,
            resolve_cross_refs=True
        ))
    ]

    results = []

    for name, config in configurations:
        print(f"\n--- Testing configuration: {name} ---")
        indexer = DocumentIndexer(config)

        start_time = time.time()
        doc_index = await indexer.index(
            text=SAMPLE_10K[:1000],  # Use smaller sample for speed
            doc_name=f"Test_{name}"
        )
        elapsed = time.time() - start_time

        stats = analyze_tree_structure(doc_index)
        results.append({
            "name": name,
            "time": elapsed,
            "nodes": stats["total_nodes"],
            "summaries": stats["nodes_with_summaries"],
            "metadata": stats["nodes_with_metadata"],
            "crossrefs": stats["nodes_with_crossrefs"]
        })

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Nodes: {stats['total_nodes']}")
        print(f"  With summaries: {stats['nodes_with_summaries']}")
        print(f"  With metadata: {stats['nodes_with_metadata']}")
        print(f"  With cross-refs: {stats['nodes_with_crossrefs']}")

    # Compare results
    print("\nConfiguration Comparison Summary:")
    print(f"{'Config':<20} {'Time(s)':<10} {'Nodes':<10} {'Summaries':<12} {'Metadata':<12} {'CrossRefs':<12}")
    print("-" * 76)
    for r in results:
        print(f"{r['name']:<20} {r['time']:<10.2f} {r['nodes']:<10} {r['summaries']:<12} {r['metadata']:<12} {r['crossrefs']:<12}")


async def example3_document_types(llm_client):
    """Example 3: Index different document types"""
    print_section_header("EXAMPLE 3: Different Document Types")

    documents = [
        ("SEC 10-K", SAMPLE_10K[:1500], "TechCorp_10K"),
        ("Earnings Call", SAMPLE_EARNINGS_CALL, "TechCorp_Earnings_Q4"),
        ("Research Report", SAMPLE_RESEARCH_REPORT, "TechCorp_Research")
    ]

    config = IndexerConfig(
        llm_config=llm_client.config,
        generate_summaries=True,
        extract_metadata=True
    )
    indexer = DocumentIndexer(config)

    for doc_type, content, name in documents:
        print(f"\n--- Indexing {doc_type} ---")

        doc_index = await indexer.index(text=content, doc_name=name)

        print(f"Document type detected: {doc_index.doc_type.value}")
        print(f"Structure preview (depth 2):")
        print_tree_with_details(
            doc_index.structure,
            doc_index,
            max_depth=2,
            show_summaries=False,
            show_metadata=False,
            show_crossrefs=False
        )

        # Show unique features
        if doc_index.doc_type.value == "EARNINGS_CALL":
            print("\nEarnings call specific features:")
            print("- Q&A section detected")
            print("- Speaker attribution preserved")
        elif doc_index.doc_type.value == "10-K":
            print("\nSEC filing specific features:")
            print("- Standard sections (PART I, PART II) identified")
            print("- Cross-references to notes detected")


async def example4_cross_reference_analysis(llm_client):
    """Example 4: Analyze cross-references"""
    print_section_header("EXAMPLE 4: Cross-Reference Analysis")

    config = IndexerConfig(
        llm_config=llm_client.config,
        generate_summaries=False,  # Skip summaries for speed
        resolve_cross_refs=True
    )

    indexer = DocumentIndexer(config)
    doc_index = await indexer.index(text=SAMPLE_10K, doc_name="CrossRef_Analysis")

    print("\nCross-Reference Network:")

    # Collect all cross-references
    all_refs = []

    def collect_refs(node: TreeNode, path: str = ""):
        current_path = f"{path}/{node.title}" if path else node.title
        for ref in node.cross_references:
            all_refs.append({
                "from_node": node.node_id,
                "from_title": node.title,
                "reference_text": ref.reference_text,
                "target_id": ref.target_node_id,
                "target_description": ref.target_description,
                "resolved": ref.resolved
            })
        for child in node.children:
            collect_refs(child, current_path)

    for root in doc_index.structure:
        collect_refs(root)

    print(f"\nFound {len(all_refs)} cross-references:")
    for ref in all_refs:
        target = doc_index.find_node(ref['target_id']) if ref['target_id'] else None
        target_title = target.title if target else "Unknown"
        print(f"  [{ref['from_node']}] {ref['from_title']}")
        print(f"    → \"{ref['reference_text']}\" links to [{ref['target_id']}] {target_title}")


async def example5_performance_optimization(llm_client):
    """Example 5: Performance optimization techniques"""
    print_section_header("EXAMPLE 5: Performance Optimization")

    # Test different chunk sizes
    chunk_sizes = [256, 512, 1024]

    print("Testing different chunk token sizes:")
    for chunk_size in chunk_sizes:
        config = IndexerConfig(
            llm_config=llm_client.config,
            chunk_config=ChunkConfig(max_chunk_tokens=chunk_size),
            generate_summaries=False,  # Disable for performance testing
            extract_metadata=False,
            max_concurrent_summaries=5
        )

        indexer = DocumentIndexer(config)

        start_time = time.time()
        doc_index = await indexer.index(
            text=SAMPLE_10K,
            doc_name=f"Perf_Test_{chunk_size}"
        )
        elapsed = time.time() - start_time

        print(f"\n  Max chunk tokens: {chunk_size}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Total chunks: {len(doc_index.chunks)}")
        print(f"  Chunks per second: {len(doc_index.chunks)/elapsed:.1f}")

    print("\n💡 Performance Tips:")
    print("1. Larger chunk sizes = fewer LLM calls but less granular structure")
    print("2. Disable features you don't need (summaries, metadata)")
    print("3. Use caching for repeated indexing of similar documents")
    print("4. Increase max_concurrent_summaries for faster parallel processing")


async def main():
    """Run all examples"""
    print("\n" + "#" * 60)
    print("DocumentIndexer Deep Dive")
    print("#" * 60)

    # Load environment
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)

    # Create LLM client
    import os
    provider = os.getenv("DOCUMENTINDEX_LLM_PROVIDER", "bedrock").lower()

    if provider == "bedrock":
        llm_client = create_bedrock_client(
            model=os.getenv("DOCUMENTINDEX_LLM_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
            region=os.getenv("AWS_REGION", "us-east-1")
        )
    elif provider == "azure":
        llm_client = create_azure_client(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    print(f"\nUsing LLM: {provider} / {llm_client.config.model}")

    # Run examples
    await example1_basic_indexing(llm_client)
    await example2_configuration_comparison(llm_client)
    await example3_document_types(llm_client)
    await example4_cross_reference_analysis(llm_client)
    await example5_performance_optimization(llm_client)

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())