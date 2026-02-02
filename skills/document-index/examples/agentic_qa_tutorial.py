"""
AgenticQA Tutorial - Intelligent question answering with reasoning traces

This example demonstrates:
1. Simple factual questions with step-by-step reasoning
2. Complex analytical questions requiring synthesis
3. Multi-hop reasoning across document sections
4. Citation generation with source excerpts
5. Confidence scoring and calibration
6. Streaming responses for real-time interaction
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from documentindex import (
    DocumentIndexer,
    AgenticQA,
    IndexerConfig,
    AgenticQAConfig,
    QAResult,
    Citation,
    create_azure_client,
    create_bedrock_client,
)

# Comprehensive document for Q&A testing
SAMPLE_DOCUMENT = """TECHCORP INC.
2024 ANNUAL REPORT

EXECUTIVE SUMMARY
TechCorp achieved record financial performance in 2024 with revenue of $15.2 billion,
representing 15% year-over-year growth. Our cloud platform division led growth at 25%,
while maintaining strong profitability with operating margins of 22%.

PART I - BUSINESS OVERVIEW

COMPANY HISTORY AND MISSION
Founded in 2010 by Jane Smith and John Doe, TechCorp began as a small startup focused
on cloud storage solutions. Our mission is to democratize access to enterprise-grade
technology for businesses of all sizes.

Key milestones:
- 2010: Company founded with initial funding of $2 million
- 2012: Launched CloudPlatform 1.0
- 2015: IPO at $25 per share, raising $500 million
- 2018: Acquired DataAnalytics Corp for $1.2 billion
- 2020: Reached 10,000 customers
- 2024: Exceeded 50,000 customers and $15 billion revenue

PRODUCTS AND SERVICES

1. CloudPlatform (65% of revenue)
Our flagship infrastructure-as-a-service offering provides:
- Scalable compute and storage resources
- 99.99% uptime SLA
- Global presence with 45 data centers
- AI-optimized infrastructure

Pricing: Starting at $100/month for small businesses, enterprise pricing varies

2. EnterpriseSuite (25% of revenue)
Comprehensive business management software including:
- ERP and financial management
- Human resources and payroll
- Supply chain management
- Customer relationship management

Average contract value: $250,000 annually

3. DataAnalytics (10% of revenue)
Advanced analytics and machine learning platform:
- Real-time data processing
- Pre-built ML models
- Custom algorithm development
- Visualization tools

Growth rate: 45% YoY despite being smallest segment

COMPETITIVE LANDSCAPE

We compete primarily with:
- Amazon Web Services (market leader with 32% share)
- Microsoft Azure (23% market share)
- Google Cloud Platform (10% market share)
- TechCorp (8% market share)

Our competitive advantages:
1. Integrated platform reducing complexity
2. Superior customer support (NPS score of 72)
3. Industry-specific solutions
4. Competitive pricing with no egress fees

PART II - FINANCIAL PERFORMANCE

REVENUE ANALYSIS

Total Revenue: $15.2 billion (+15% YoY)

By Product:
- CloudPlatform: $9.88 billion (+25% YoY)
- EnterpriseSuite: $3.80 billion (+8% YoY)
- DataAnalytics: $1.52 billion (+45% YoY)

By Geography:
- Americas: $9.12 billion (60%)
- EMEA: $3.80 billion (25%)
- APAC: $2.28 billion (15%)

By Customer Size:
- Enterprise (>$1M ARR): $7.6 billion (50%)
- Mid-market ($100K-$1M): $4.56 billion (30%)
- SMB (<$100K): $3.04 billion (20%)

Q4 2024 Performance:
- Q4 Revenue: $4.1 billion (+18% YoY)
- New customers added: 3,500
- Net revenue retention: 115%

PROFITABILITY METRICS

Gross Profit: $10.34 billion (68% margin)
- CloudPlatform gross margin: 75%
- EnterpriseSuite gross margin: 60%
- DataAnalytics gross margin: 55%

Operating Income: $3.34 billion (22% margin)
- R&D expenses: $2.28 billion (15% of revenue)
- Sales & Marketing: $3.04 billion (20% of revenue)
- G&A: $1.52 billion (10% of revenue)

Net Income: $2.43 billion (16% margin)
Earnings Per Share: $4.86 (diluted)

Cash Flow:
- Operating cash flow: $4.1 billion
- Free cash flow: $3.5 billion
- Cash and investments: $8.2 billion

CUSTOMER METRICS

Total Customers: 52,000 (+15% YoY)
- Enterprise customers: 500 (+25% YoY)
- Average revenue per customer: $292,000

Customer Retention:
- Gross retention: 95%
- Net revenue retention: 115%
- Churn rate: 5% (improved from 7% in 2023)

Customer Satisfaction:
- Net Promoter Score: 72 (industry average: 45)
- Support ticket resolution time: 4 hours average
- Customer success team: 1,200 employees

PART III - STRATEGIC INITIATIVES

AI INTEGRATION ROADMAP

We are investing $1 billion annually in AI capabilities:

1. AI-Powered Features (Launching 2025)
- Automated infrastructure optimization
- Predictive analytics in EnterpriseSuite
- Natural language querying in DataAnalytics
- Intelligent security threat detection

2. GenAI Platform (Beta 2025, GA 2026)
- Large language model hosting
- Fine-tuning capabilities
- RAG (Retrieval Augmented Generation) tools
- Industry-specific AI models

Expected impact: 5-10% revenue uplift by 2026

INTERNATIONAL EXPANSION

APAC Focus:
- Opening 3 new data centers in 2025 (Singapore, Tokyo, Sydney)
- Hiring 500 local employees
- Partnering with regional system integrators
- Target: Reach 25% revenue from APAC by 2027

European Growth:
- GDPR-compliant sovereign cloud offering
- Local language support for all products
- Partnerships with European telcos

SUSTAINABILITY INITIATIVES

Environmental Goals:
- Carbon neutral by 2030
- 100% renewable energy for data centers by 2027
- 30% reduction in water usage by 2026

Current Progress:
- 65% renewable energy usage (up from 45% in 2023)
- 20% reduction in carbon emissions YoY
- LEED certified facilities: 80%

Social Impact:
- $50 million in cloud credits for nonprofits
- Free training programs reached 100,000 individuals
- 45% of new hires from underrepresented groups

PART IV - RISK FACTORS

OPERATIONAL RISKS

1. Cybersecurity Threats
- Increasing sophistication of attacks
- Potential for data breaches affecting customer trust
- Regulatory penalties for security failures
- Mitigation: $200M annual security investment, bug bounty program

2. Service Disruptions
- Dependence on third-party infrastructure
- Natural disasters affecting data centers
- Software bugs causing outages
- Mitigation: Multi-region redundancy, disaster recovery plans

3. Talent Competition
- Shortage of skilled engineers
- High compensation pressure (average engineer salary: $250,000)
- Competition from FAANG companies
- Mitigation: Stock compensation, remote work options, learning programs

MARKET RISKS

1. Economic Sensitivity
- IT spending cuts during recessions
- Foreign exchange exposure (40% non-USD revenue)
- Interest rate impact on growth investments
- Customer concentration (top 10 = 15% of revenue)

2. Competitive Pressure
- Price wars with larger competitors
- Technology commoditization
- Open source alternatives
- Platform lock-in concerns

REGULATORY RISKS

1. Data Privacy Regulations
- GDPR fines up to 4% of global revenue
- CCPA and state privacy laws
- Cross-border data transfer restrictions
- Industry-specific compliance (HIPAA, PCI-DSS)

2. Antitrust Scrutiny
- Potential breakup discussions
- Acquisition approval challenges
- Bundling practice investigations

APPENDIX A - FINANCIAL TABLES

[Detailed financial statements would appear here]

APPENDIX B - GLOSSARY

ARR - Annual Recurring Revenue
SaaS - Software as a Service
NPS - Net Promoter Score
GDPR - General Data Protection Regulation
"""


def print_section_header(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def format_reasoning_step(step: dict, step_num: int) -> str:
    """Format a reasoning step for display"""
    action_emojis = {
        "plan": "📋",
        "read_section": "📖",
        "follow_reference": "🔗",
        "search": "🔍",
        "synthesize": "🧩",
        "answer": "✅",
        "give_up": "❌"
    }

    action = step.get('action', 'unknown')
    emoji = action_emojis.get(action, "▶️")
    output = f"\nStep {step_num} [{action}] {emoji}"

    if step.get('node_id'):
        output += f" - Node: {step.get('node_id')}"

    output += f"\n  Reasoning: {step.get('reasoning', '')}"

    if step.get('findings'):
        output += f"\n  Findings: {step.get('findings')}"

    return output


def format_confidence_score(confidence: float) -> str:
    """Format confidence score with interpretation"""
    if confidence >= 0.9:
        interpretation = "Very High - Found comprehensive direct evidence"
    elif confidence >= 0.8:
        interpretation = "High - Found clear relevant information"
    elif confidence >= 0.7:
        interpretation = "Moderate - Found partial information"
    elif confidence >= 0.6:
        interpretation = "Low - Limited or indirect evidence"
    else:
        interpretation = "Very Low - Minimal supporting evidence"

    return f"{confidence:.2f} ({interpretation})"


def display_qa_result(question: str, result: QAResult, show_full_trace: bool = True):
    """Display Q&A result with formatting"""
    print(f"\n❓ Question: {question}")
    print("-" * 60)

    # Show reasoning trace
    if show_full_trace and result.reasoning_trace:
        print("\n🧠 Reasoning Trace:")
        for i, step in enumerate(result.reasoning_trace, 1):
            print(format_reasoning_step(step, i))

    # Show answer
    print(f"\n✨ Answer:")
    print(result.answer)

    # Show confidence
    print(f"\n📊 Confidence: {format_confidence_score(result.confidence)}")

    # Show citations
    if result.citations:
        print(f"\n📚 Citations ({len(result.citations)}):")
        for i, citation in enumerate(result.citations, 1):
            print(f"\n  {i}. [{citation.node_id}] {citation.node_title}")
            if citation.excerpt:
                excerpt = citation.excerpt[:200] + "..." if len(citation.excerpt) > 200 else citation.excerpt
                print(f"     \"{excerpt}\"")

    # Show nodes visited
    if result.nodes_visited:
        print(f"\n🗺️ Nodes visited: {', '.join(result.nodes_visited)}")


async def example1_factual_questions(doc_index, llm_client):
    """Example 1: Simple factual questions with clear answers"""
    print_section_header("EXAMPLE 1: Factual Questions")

    qa = AgenticQA(doc_index, llm_client=llm_client)

    factual_questions = [
        "What was TechCorp's total revenue in 2024?",
        "Who founded TechCorp and when?",
        "How many customers does TechCorp have?",
        "What is the company's gross margin?"
    ]

    print("\nDemonstrating factual question answering:")

    for question in factual_questions:
        result = await qa.answer(question)
        display_qa_result(question, result, show_full_trace=False)
        print("\n" + "-" * 40)

    # Show one with full trace
    print("\n🔍 Detailed trace for a factual question:")
    question = "What are TechCorp's three main product lines?"
    result = await qa.answer(question)
    display_qa_result(question, result, show_full_trace=True)


async def example2_analytical_questions(doc_index, llm_client):
    """Example 2: Analytical questions requiring synthesis"""
    print_section_header("EXAMPLE 2: Analytical Questions")

    qa = AgenticQA(doc_index, llm_client=llm_client)

    analytical_questions = [
        "How has TechCorp's profitability changed over time?",
        "What factors are driving revenue growth?",
        "Compare the growth rates of different product segments."
    ]

    print("\nDemonstrating analytical reasoning:")

    for i, question in enumerate(analytical_questions):
        print(f"\n{'='*60}")
        print(f"Analysis {i+1}:")

        result = await qa.answer(question)
        display_qa_result(question, result, show_full_trace=True)

        # Highlight the synthesis process
        synthesis_steps = [s for s in result.reasoning_trace if s.get('action') in ['synthesize', 'answer']]
        if synthesis_steps:
            print("\n💡 Key Synthesis Steps:")
            for step in synthesis_steps:
                print(f"  - {step.get('findings') or step.get('reasoning', '')}")


async def example3_multi_hop_reasoning(doc_index, llm_client):
    """Example 3: Questions requiring multiple hops"""
    print_section_header("EXAMPLE 3: Multi-Hop Reasoning")

    config = AgenticQAConfig(
        max_iterations=10,  # Allow more steps for complex reasoning
        follow_cross_refs=True,
        confidence_threshold=0.8
    )

    qa = AgenticQA(doc_index, llm_client=llm_client)

    multi_hop_questions = [
        "What percentage of revenue comes from the fastest growing geographic region?",
        "How does TechCorp's customer retention compare to its NPS score?",
        "What risks could impact the company's AI integration roadmap?"
    ]

    print("\nDemonstrating multi-hop reasoning:")

    for question in multi_hop_questions[:1]:  # Show one in detail
        result = await qa.answer(question, config)
        display_qa_result(question, result, show_full_trace=True)

        # Analyze the reasoning path
        print("\n🛤️ Reasoning Path Analysis:")
        sections_visited = []
        for step in result.reasoning_trace:
            if step.get('action') == "read_section" and step.get('node_id'):
                node = doc_index.find_node(step.get('node_id'))
                if node:
                    sections_visited.append(node.title)

        print(f"  Sections traversed ({len(sections_visited)}):")
        for i, section in enumerate(sections_visited, 1):
            print(f"    {i}. {section}")


async def example4_comparative_analysis(doc_index, llm_client):
    """Example 4: Comparative and relationship questions"""
    print_section_header("EXAMPLE 4: Comparative Analysis")

    qa = AgenticQA(doc_index, llm_client=llm_client)

    comparative_questions = [
        "Compare Q4 performance to the full year 2024 results.",
        "How does CloudPlatform's margin compare to other products?",
        "What's the relationship between R&D spending and revenue growth?"
    ]

    print("\nDemonstrating comparative analysis:")

    for question in comparative_questions:
        result = await qa.answer(question)

        # Show condensed results
        print(f"\n❓ {question}")
        print(f"💬 {result.answer}")
        print(f"📊 Confidence: {result.confidence:.2f}")
        print(f"📍 Sources: {len(result.citations)} citations from {len(set(c.node_id for c in result.citations))} sections")


async def example5_confidence_calibration(doc_index, llm_client):
    """Example 5: Understanding confidence scoring"""
    print_section_header("EXAMPLE 5: Confidence Calibration")

    qa = AgenticQA(doc_index, llm_client=llm_client)

    # Questions with varying expected confidence levels
    confidence_test_questions = [
        ("What was the exact revenue in 2024?", "High - Specific fact in document"),
        ("What will revenue be in 2025?", "Low - Future projection with uncertainty"),
        ("Who are TechCorp's main competitors?", "High - Clearly listed"),
        ("Will TechCorp acquire any companies in 2025?", "Very Low - Speculation"),
        ("What is TechCorp's environmental impact?", "Moderate - Partial information")
    ]

    print("\nTesting confidence calibration across different question types:")
    print(f"\n{'Question':<50} {'Confidence':<10} {'Expected':<30}")
    print("-" * 90)

    results = []
    for question, expected in confidence_test_questions:
        result = await qa.answer(question)
        results.append((question, result.confidence, expected))
        print(f"{question:<50} {result.confidence:<10.2f} {expected:<30}")

    # Explain confidence factors
    print("\n📊 Confidence Factors:")
    print("  1. Directness of evidence (explicit vs. inferred)")
    print("  2. Completeness of information")
    print("  3. Number of supporting sources")
    print("  4. Recency and relevance of data")
    print("  5. Ambiguity in the question")


async def example6_streaming_qa(doc_index, llm_client):
    """Example 6: Streaming responses for real-time interaction"""
    print_section_header("EXAMPLE 6: Streaming Q&A Responses")

    qa = AgenticQA(doc_index, llm_client=llm_client)

    question = "Provide a comprehensive analysis of TechCorp's financial health and growth prospects."

    print(f"\n❓ Question: {question}")
    print("\n🌊 Streaming answer:")
    print("-" * 60)

    # Get streaming result
    result = await qa.answer_stream(question)

    # Stream the answer as it's generated
    print("\n✨ Answer (streaming):")
    full_answer = ""
    async for chunk in result.answer_stream:
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_answer += chunk.content
        if chunk.is_complete:
            break

    # Show final metadata
    print(f"\n\n📊 Final Confidence: {result.confidence:.2f}")
    print(f"📚 Citations: {len(result.citations)} sources")

    # Show reasoning trace after completion
    if result.reasoning_trace:
        print(f"\n🧠 Reasoning Steps Taken: {len(result.reasoning_trace)}")
    print(f"⏱️ Nodes explored: {len(result.nodes_visited)}")


async def example7_custom_configuration(doc_index, llm_client):
    """Example 7: Custom Q&A configurations"""
    print_section_header("EXAMPLE 7: Custom Configuration Options")

    configurations = [
        ("Quick Mode", AgenticQAConfig(
            max_iterations=3,
            confidence_threshold=0.6,
            max_context_tokens=2000
        )),
        ("Thorough Mode", AgenticQAConfig(
            max_iterations=15,
            confidence_threshold=0.9,
            max_context_tokens=8000,
            follow_cross_refs=True
        )),
        ("Citation Focus", AgenticQAConfig(
            generate_citations=True,
            max_context_tokens=4000
        ))
    ]

    question = "What are TechCorp's main competitive advantages?"

    print(f"\nTesting different configurations for: \"{question}\"")

    for config_name, config in configurations:
        print(f"\n--- {config_name} ---")
        qa = AgenticQA(doc_index, llm_client=llm_client)

        start_time = time.time()
        result = await qa.answer(question, config)
        elapsed = time.time() - start_time

        print(f"Time: {elapsed:.2f}s")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Steps taken: {len(result.reasoning_trace)}")
        print(f"Citations: {len(result.citations)}")
        if result.citations and config.generate_citations:
            print(f"Excerpt length: {sum(len(c.excerpt or '') for c in result.citations)} chars")


async def create_indexed_document(llm_client):
    """Create an indexed document for Q&A"""
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
        doc_name="TechCorp_Annual_Report_2024"
    )

    print(f"✓ Document indexed: {doc_index.get_node_count()} nodes")
    return doc_index


async def main():
    """Run all examples"""
    print("\n" + "#" * 60)
    print("AgenticQA Tutorial")
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
    await example1_factual_questions(doc_index, llm_client)
    await example2_analytical_questions(doc_index, llm_client)
    await example3_multi_hop_reasoning(doc_index, llm_client)
    await example4_comparative_analysis(doc_index, llm_client)
    await example5_confidence_calibration(doc_index, llm_client)
    await example6_streaming_qa(doc_index, llm_client)
    await example7_custom_configuration(doc_index, llm_client)

    print("\n" + "=" * 60)
    print("Tutorial completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())