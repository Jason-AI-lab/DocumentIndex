"""
NodeSearcher Showcase - Intelligent document search demonstrations

This example demonstrates:
1. Basic relevance scoring with explanations
2. Understanding match reasoning
3. Extracting relevant excerpts
4. Batch search operations for multiple topics
5. Cross-reference expansion in search
6. Search result analysis and optimization
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

from dotenv import load_dotenv
from documentindex import (
    DocumentIndexer,
    NodeSearcher,
    IndexerConfig,
    NodeSearchConfig,
    NodeMatch,
    create_azure_client,
    create_bedrock_client,
)

# Comprehensive financial document for testing
SAMPLE_DOCUMENT = """FORM 10-K
ANNUAL REPORT - TECHCORP INC.

PART I

ITEM 1. BUSINESS

Company Overview
TechCorp Inc. is a leading provider of cloud computing and artificial intelligence
solutions. Founded in 2010, we have grown to become a major player in the enterprise
software market with over 50,000 customers globally.

Our Revenue Model
We generate revenue through three primary channels:
1. Subscription Services - Recurring SaaS revenue representing 70% of total revenue
2. Professional Services - Implementation and consulting (20% of revenue)
3. License Sales - Traditional software licenses (10% of revenue)

In fiscal 2024, total revenue reached $15.2 billion, a 15% increase year-over-year.
Cloud subscription revenue grew 25% to $10.6 billion, driven by strong demand for
our AI-powered analytics platform.

Market Position
We compete with major technology companies including AWS, Microsoft Azure, and Google
Cloud. Our competitive advantages include:
- Integrated AI capabilities across all products
- Superior customer support with 24/7 availability
- Industry-leading security certifications
- Flexible deployment options (cloud, on-premise, hybrid)

ITEM 1A. RISK FACTORS

Cybersecurity Risks
As a cloud service provider, we face significant cybersecurity threats. A major
data breach could result in:
- Loss of customer trust and business
- Regulatory fines and legal liability
- Damage to our reputation and brand
- Increased security costs

We invest heavily in security measures including encryption, multi-factor
authentication, and regular security audits. See Item 1C for our security framework.

Competition Risks
The cloud computing market is intensely competitive. Key risks include:
- Price pressure from larger competitors
- Rapid technological change requiring constant innovation
- Customer concentration with our top 10 customers representing 35% of revenue
- Difficulty attracting and retaining technical talent

Market and Economic Risks
Our business is sensitive to economic conditions:
- Economic downturns reduce IT spending
- Foreign exchange fluctuations impact international revenue (40% of total)
- Interest rate changes affect our cost of capital
- Supply chain disruptions can delay hardware deployments

Regulatory and Compliance Risks
We operate in a highly regulated environment:
- Data privacy laws (GDPR, CCPA) require significant compliance efforts
- Export controls limit our ability to serve certain markets
- Tax law changes could increase our effective tax rate
- Environmental regulations affect our data center operations

ITEM 1C. CYBERSECURITY

Security Framework
Our comprehensive security program includes:
- ISO 27001 and SOC 2 Type II certifications
- Zero-trust architecture implementation
- AI-powered threat detection and response
- Regular penetration testing by third parties
- Employee security training programs

Data Protection Measures
- End-to-end encryption for data in transit and at rest
- Geographic data redundancy across multiple regions
- Automated backup systems with point-in-time recovery
- Access controls with principle of least privilege

PART II

ITEM 7. MANAGEMENT'S DISCUSSION AND ANALYSIS

Financial Performance Overview
Fiscal 2024 was a record year with strong growth across all segments:

Revenue Analysis:
- Total revenue: $15.2 billion (+15% YoY)
- Cloud Platform: $10.6 billion (+25% YoY)
- Professional Services: $3.0 billion (+5% YoY)
- License Sales: $1.6 billion (-10% YoY)

The shift from license sales to subscription revenue continues as planned,
improving our revenue predictability and customer lifetime value.

Profitability Metrics:
- Gross margin: 68% (up from 65% in 2023)
- Operating margin: 22% (up from 20% in 2023)
- Net income: $2.8 billion (+30% YoY)
- Free cash flow: $3.5 billion (+35% YoY)

Geographic Performance:
- Americas: $9.1 billion (60% of total)
- EMEA: $3.8 billion (25% of total)
- APAC: $2.3 billion (15% of total)

APAC showed the strongest growth at 35% YoY, driven by expansion in Japan
and Southeast Asia markets.

Customer Metrics:
- Total customers: 52,000 (+15% YoY)
- Enterprise customers (>$1M ARR): 500 (+25% YoY)
- Net revenue retention: 115%
- Customer churn rate: 5% (improved from 7% in 2023)

ITEM 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES

Market Risk Exposures:
- Foreign currency risk: 40% of revenue in non-USD currencies
- Interest rate risk: $2 billion in variable rate debt
- Credit risk: Accounts receivable of $1.8 billion

We use hedging strategies to mitigate foreign exchange and interest rate risks.
See Note 12 for derivative instruments details.

NOTE 12. FINANCIAL INSTRUMENTS AND RISK MANAGEMENT

We utilize various financial instruments to manage risk:
- Foreign exchange forwards to hedge currency exposure
- Interest rate swaps to fix borrowing costs
- Credit insurance for major customer accounts

NOTE 15. SEGMENT INFORMATION

Detailed revenue breakdown by product and geography:

Product Segments:
- Cloud Platform: Revenue $10.6B, Operating margin 35%
- Professional Services: Revenue $3.0B, Operating margin 15%
- License Sales: Revenue $1.6B, Operating margin 45%

Investment priorities for 2025:
- AI and machine learning capabilities
- Geographic expansion in APAC
- Enhanced security features
- Developer ecosystem growth
"""


def print_section_header(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def format_relevance_score(score: float) -> str:
    """Format relevance score with visual indicator"""
    if score >= 0.9:
        indicator = "🟢"
        label = "HIGHLY RELEVANT"
    elif score >= 0.7:
        indicator = "🟡"
        label = "RELEVANT"
    elif score >= 0.5:
        indicator = "🟠"
        label = "SOMEWHAT RELEVANT"
    else:
        indicator = "🔴"
        label = "LOW RELEVANCE"

    return f"{indicator} {score:.2f} ({label})"


def display_search_results(
    query: str,
    matches: List[NodeMatch],
    doc_index,
    show_excerpts: bool = True,
    show_reasoning: bool = True,
    max_results: int = 5
):
    """Display search results with formatting"""
    print(f"\n🔍 Search Query: \"{query}\"")
    print(f"Found {len(matches)} matches")

    if not matches:
        print("No relevant matches found.")
        return

    print("\nTop Results:")
    print("-" * 60)

    for i, match in enumerate(matches[:max_results]):
        node = match.node
        print(f"\n{i+1}. [{node.node_id}] {node.title}")
        print(f"   Relevance: {format_relevance_score(match.relevance_score)}")

        if show_reasoning and match.match_reason:
            print(f"   Reasoning: {match.match_reason}")

        if show_excerpts and match.matched_excerpts:
            print(f"   Excerpts:")
            for excerpt in match.matched_excerpts[:2]:  # Show max 2 excerpts
                # Truncate long excerpts
                text = excerpt.text
                if len(text) > 150:
                    text = text[:147] + "..."
                print(f"     \"...{text}...\"")

        # Show node context
        if node.parent_id:
            parent = doc_index.find_node(node.parent_id)
            if parent:
                print(f"   Context: Part of {parent.title}")


async def example1_basic_relevance_scoring(doc_index, llm_client):
    """Example 1: Basic relevance scoring with detailed explanations"""
    print_section_header("EXAMPLE 1: Relevance Scoring Explained")

    searcher = NodeSearcher(doc_index, llm_client=llm_client)

    # Test queries with different expected relevance levels
    test_queries = [
        "revenue growth and financial performance",
        "cybersecurity threats and data protection",
        "employee benefits and compensation",  # Not in document
        "cloud computing market competition",
        "quantum computing advances"  # Not in document
    ]

    print("\nTesting relevance scoring across different queries:")

    for query in test_queries:
        matches = await searcher.find_related_nodes(query)
        display_search_results(query, matches, doc_index, max_results=3)

    # Explain scoring
    print("\n📊 Relevance Scoring Guide:")
    print("  0.9-1.0 (🟢): Direct match, primary topic of the section")
    print("  0.7-0.8 (🟡): Strong relevance, contains substantial related content")
    print("  0.5-0.6 (🟠): Moderate relevance, mentions the topic")
    print("  0.0-0.4 (🔴): Low relevance, tangential or no relation")


async def example2_batch_search_operations(doc_index, llm_client):
    """Example 2: Batch search for multiple topics"""
    print_section_header("EXAMPLE 2: Batch Search Operations")

    searcher = NodeSearcher(doc_index, llm_client=llm_client)

    # Define multiple search topics
    topics = {
        "Financial Metrics": "revenue, profit, margins, cash flow",
        "Risk Factors": "risks, threats, challenges, vulnerabilities",
        "Growth Strategy": "growth, expansion, investment, market opportunity",
        "Technology": "AI, cloud, platform, innovation, technology"
    }

    print("\nSearching for multiple topics simultaneously...")

    all_results = {}
    start_time = time.time()

    # Perform searches
    for category, query in topics.items():
        matches = await searcher.find_related_nodes(query)
        all_results[category] = matches
        print(f"  ✓ {category}: Found {len(matches)} matches")

    elapsed = time.time() - start_time
    print(f"\nBatch search completed in {elapsed:.2f} seconds")

    # Analyze coverage
    print("\n📈 Topic Coverage Analysis:")
    covered_nodes = set()
    for category, matches in all_results.items():
        category_nodes = {m.node.node_id for m in matches if m.relevance_score >= 0.7}
        covered_nodes.update(category_nodes)
        print(f"  {category}: {len(category_nodes)} highly relevant sections")

    total_nodes = doc_index.get_node_count()
    coverage_pct = (len(covered_nodes) / total_nodes) * 100
    print(f"\nTotal coverage: {len(covered_nodes)}/{total_nodes} nodes ({coverage_pct:.1f}%)")

    # Show topic overlap
    print("\n🔄 Topic Overlap Matrix:")
    categories = list(topics.keys())
    for i, cat1 in enumerate(categories):
        nodes1 = {m.node.node_id for m in all_results[cat1] if m.relevance_score >= 0.7}
        overlaps = []
        for j, cat2 in enumerate(categories):
            if i < j:
                nodes2 = {m.node.node_id for m in all_results[cat2] if m.relevance_score >= 0.7}
                overlap = len(nodes1.intersection(nodes2))
                if overlap > 0:
                    overlaps.append(f"{cat2} ({overlap})")
        if overlaps:
            print(f"  {cat1} overlaps with: {', '.join(overlaps)}")


async def example3_cross_reference_search(doc_index, llm_client):
    """Example 3: Search with cross-reference expansion"""
    print_section_header("EXAMPLE 3: Cross-Reference Expansion")

    # First, do a regular search
    config = NodeSearchConfig(
        relevance_threshold=0.7,
        max_results=10,
        follow_cross_refs=False
    )
    searcher = NodeSearcher(doc_index, llm_client=llm_client)

    query = "segment information and geographic revenue"
    print(f"\n1️⃣ Regular search for: \"{query}\"")

    regular_matches = await searcher.find_related_nodes(query, config)
    regular_ids = {m.node.node_id for m in regular_matches}
    print(f"Found {len(regular_matches)} direct matches")

    # Now search with cross-reference expansion
    config.follow_cross_refs = True

    print(f"\n2️⃣ Search with cross-reference expansion:")
    expanded_matches = await searcher.find_related_nodes(query, config)
    expanded_ids = {m.node.node_id for m in expanded_matches}

    # Show the difference
    additional_nodes = expanded_ids - regular_ids
    print(f"Found {len(expanded_matches)} total matches ({len(additional_nodes)} from cross-refs)")

    if additional_nodes:
        print("\n🔗 Additional nodes found through cross-references:")
        for node_id in additional_nodes:
            node = doc_index.find_node(node_id)
            if node:
                print(f"  [{node.node_id}] {node.title}")
                # Show why it was included
                for match in expanded_matches:
                    if match.node.node_id == node_id:
                        print(f"    → Included because: {match.match_reason}")
                        break


async def example4_search_result_analysis(doc_index, llm_client):
    """Example 4: Analyze and visualize search results"""
    print_section_header("EXAMPLE 4: Search Result Analysis")

    searcher = NodeSearcher(doc_index, llm_client=llm_client)

    # Comprehensive search
    query = "financial performance revenue growth profit margins"
    matches = await searcher.find_related_nodes(query)

    print(f"\n🔍 Analyzing results for: \"{query}\"")
    print(f"Total matches: {len(matches)}")

    # Score distribution
    print("\n📊 Score Distribution:")
    score_buckets = defaultdict(int)
    for match in matches:
        bucket = int(match.relevance_score * 10) / 10
        score_buckets[bucket] += 1

    for score in sorted(score_buckets.keys(), reverse=True):
        count = score_buckets[score]
        bar = "█" * count
        print(f"  {score:.1f}: {bar} ({count})")

    # Results by document section
    print("\n📑 Results by Section:")
    section_matches = defaultdict(list)
    for match in matches:
        # Find top-level parent
        current = match.node
        while current.parent_id:
            parent = doc_index.find_node(current.parent_id)
            if parent and parent.parent_id:
                current = parent
            else:
                break

        section_matches[current.title].append(match)

    for section, section_results in section_matches.items():
        avg_score = sum(m.relevance_score for m in section_results) / len(section_results)
        print(f"  {section}: {len(section_results)} matches (avg score: {avg_score:.2f})")

    # Extract key insights
    print("\n💡 Key Insights from Top Matches:")
    top_matches = sorted(matches, key=lambda m: m.relevance_score, reverse=True)[:5]

    insights = []
    for match in top_matches:
        if match.matched_excerpts:
            # Extract numeric values from excerpts
            import re
            for excerpt in match.matched_excerpts:
                # Find percentages
                percentages = re.findall(r'\d+(?:\.\d+)?%', excerpt.text)
                # Find dollar amounts
                amounts = re.findall(r'\$[\d.]+\s*(?:billion|million)', excerpt.text)

                if percentages or amounts:
                    insights.append({
                        'node': match.node.title,
                        'values': percentages + amounts,
                        'context': excerpt.text[:100]
                    })

    for insight in insights[:5]:  # Show top 5 insights
        print(f"\n  From {insight['node']}:")
        print(f"    Key values: {', '.join(insight['values'])}")
        print(f"    Context: \"{insight['context']}...\"")


async def example5_search_optimization(doc_index, llm_client):
    """Example 5: Search performance optimization techniques"""
    print_section_header("EXAMPLE 5: Search Optimization Techniques")

    queries = [
        "revenue growth financial performance",
        "cybersecurity risk data protection",
        "cloud platform competitive advantage"
    ]

    # Test different configurations
    configurations = [
        ("Default", NodeSearchConfig()),
        ("High Threshold", NodeSearchConfig(relevance_threshold=0.8, max_results=5)),
        ("Low Threshold", NodeSearchConfig(relevance_threshold=0.3, max_results=20)),
        ("No Cross-Refs", NodeSearchConfig(follow_cross_refs=False)),
        ("Cached", NodeSearchConfig(relevance_threshold=0.7))  # Will benefit from caching
    ]

    print("\nTesting different search configurations:")

    for config_name, config in configurations:
        print(f"\n--- Configuration: {config_name} ---")
        if config_name != "Default":
            print(f"Settings: threshold={config.relevance_threshold}, "
                  f"max_results={config.max_results}, "
                  f"expand_refs={config.follow_cross_refs}")

        searcher = NodeSearcher(doc_index, llm_client=llm_client)

        total_time = 0
        total_results = 0

        for query in queries:
            start = time.time()
            matches = await searcher.find_related_nodes(query, config)
            elapsed = time.time() - start

            total_time += elapsed
            total_results += len(matches)

        avg_time = total_time / len(queries)
        avg_results = total_results / len(queries)

        print(f"  Avg time per search: {avg_time:.3f}s")
        print(f"  Avg results returned: {avg_results:.1f}")

    print("\n💡 Optimization Tips:")
    print("1. Higher relevance threshold = fewer results but faster")
    print("2. Disable cross-ref expansion for speed when not needed")
    print("3. Use caching for repeated similar queries")
    print("4. Limit max_results to avoid processing irrelevant matches")
    print("5. Batch similar queries together for efficiency")


async def example6_semantic_search_examples(doc_index, llm_client):
    """Example 6: Advanced semantic search examples"""
    print_section_header("EXAMPLE 6: Semantic Search Capabilities")

    searcher = NodeSearcher(doc_index, llm_client=llm_client)

    # Demonstrate semantic understanding
    semantic_tests = [
        {
            "description": "Synonym Recognition",
            "queries": [
                "revenue earnings income",
                "profit margins profitability",
                "dangers risks threats"
            ]
        },
        {
            "description": "Concept Understanding",
            "queries": [
                "company growth expansion",
                "financial stability strength",
                "market position competitive advantage"
            ]
        },
        {
            "description": "Question-Style Queries",
            "queries": [
                "how much revenue did the company make",
                "what are the main risks",
                "which geographic regions are growing fastest"
            ]
        }
    ]

    for test in semantic_tests:
        print(f"\n🧠 {test['description']}:")
        print("-" * 40)

        for query in test['queries']:
            matches = await searcher.find_related_nodes(query)
            if matches:
                top_match = matches[0]
                print(f"\nQuery: \"{query}\"")
                print(f"Best match: [{top_match.node.node_id}] {top_match.node.title}")
                print(f"Score: {format_relevance_score(top_match.relevance_score)}")
                if top_match.match_reason:
                    print(f"Why: {top_match.match_reason}")


async def create_indexed_document(llm_client):
    """Create an indexed document for examples"""
    print("\nPreparing document index...")

    config = IndexerConfig(
        llm_config=llm_client.config,
        generate_summaries=True,
        extract_metadata=True,
        resolve_cross_refs=True
    )

    indexer = DocumentIndexer(config)
    doc_index = await indexer.index(
        text=SAMPLE_DOCUMENT,
        doc_name="TechCorp_10K_Detailed"
    )

    print(f"✓ Document indexed: {doc_index.get_node_count()} nodes")
    return doc_index


async def main():
    """Run all examples"""
    print("\n" + "#" * 60)
    print("NodeSearcher Showcase")
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

    # Create indexed document
    doc_index = await create_indexed_document(llm_client)

    # Run examples
    await example1_basic_relevance_scoring(doc_index, llm_client)
    await example2_batch_search_operations(doc_index, llm_client)
    await example3_cross_reference_search(doc_index, llm_client)
    await example4_search_result_analysis(doc_index, llm_client)
    await example5_search_optimization(doc_index, llm_client)
    await example6_semantic_search_examples(doc_index, llm_client)

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())