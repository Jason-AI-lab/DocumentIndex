"""
Provenance Extraction Patterns - Exhaustive evidence discovery demonstrations

This example demonstrates:
1. Basic single-topic deep extraction with all evidence
2. Multi-category risk analysis across document
3. Evidence scoring, ranking, and threshold tuning
4. Summary generation from extracted evidence
5. Progress tracking for large document scans
6. Real-time streaming extraction
7. Export formats (JSON, CSV, and custom reports)
"""

import asyncio
import json
import csv
import time
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict

from dotenv import load_dotenv
from documentindex import (
    DocumentIndexer,
    ProvenanceExtractor,
    IndexerConfig,
    ProvenanceConfig,
    ProvenanceResult,
    NodeMatch,
    DocumentIndex,
    create_azure_client,
    create_bedrock_client,
)
from documentindex.streaming import ProgressCallback, ProgressUpdate


# ============================================================================
# Helper Functions
# ============================================================================

def print_section_header(title: str) -> None:
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def format_relevance_score(score: float) -> str:
    """Format relevance score with emoji and interpretation"""
    if score >= 0.9:
        return f"🔴 {score:.2f} (CRITICAL)"
    elif score >= 0.8:
        return f"🟠 {score:.2f} (HIGH)"
    elif score >= 0.7:
        return f"🟡 {score:.2f} (MODERATE)"
    elif score >= 0.6:
        return f"🟢 {score:.2f} (LOW)"
    else:
        return f"⚪ {score:.2f} (MINIMAL)"


def display_evidence_summary(
    result: ProvenanceResult,
    show_excerpts: bool = True,
    max_evidence: int = 10
) -> None:
    """Display provenance result with formatted output"""
    print(f"\n🎯 Topic: {result.topic}")
    print(f"📊 Scan Coverage: {result.scan_coverage*100:.0f}% ({result.total_nodes_scanned} nodes)")
    print(f"📚 Evidence Found: {len(result.evidence)} relevant sections")
    print("-" * 70)

    if result.summary:
        print(f"\n📝 Summary:")
        print(f"  {result.summary}")

    if result.evidence:
        print(f"\n🔍 Top Evidence (showing {min(max_evidence, len(result.evidence))}):")
        for i, match in enumerate(result.evidence[:max_evidence], 1):
            print(f"\n  {i}. [{match.node.node_id}] {match.node.title}")
            print(f"     Relevance: {format_relevance_score(match.relevance_score)}")
            print(f"     Reason: {match.match_reason}")

            if show_excerpts and match.matched_excerpts:
                print(f"     Key Excerpts:")
                for excerpt in match.matched_excerpts[:2]:
                    preview = excerpt[:150] + "..." if len(excerpt) > 150 else excerpt
                    print(f"       • \"{preview}\"")


def display_category_comparison(results: Dict[str, ProvenanceResult]) -> None:
    """Display comparison across multiple categories"""
    print("\n📊 Multi-Category Analysis Summary")
    print("-" * 70)
    print(f"{'Category':<30} {'Evidence':<12} {'Avg Score':<12} {'Coverage':<10}")
    print("-" * 70)

    for category, result in results.items():
        avg_score = (
            sum(m.relevance_score for m in result.evidence) / len(result.evidence)
            if result.evidence else 0
        )
        print(
            f"{category:<30} {len(result.evidence):<12} "
            f"{avg_score:<12.2f} {result.scan_coverage*100:<9.0f}%"
        )


def export_to_json(result: ProvenanceResult, filepath: Path) -> None:
    """Export provenance result to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"✅ Exported to JSON: {filepath}")


def export_to_csv(
    result: ProvenanceResult,
    doc_index: DocumentIndex,
    filepath: Path
) -> None:
    """Export evidence to CSV with columns: node_id, title, relevance_score, excerpt_count, first_excerpt"""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'node_id', 'title', 'relevance_score', 'match_reason',
            'excerpt_count', 'first_excerpt'
        ])

        for match in result.evidence:
            first_excerpt = match.matched_excerpts[0][:200] if match.matched_excerpts else ""
            writer.writerow([
                match.node.node_id,
                match.node.title,
                f"{match.relevance_score:.2f}",
                match.match_reason,
                len(match.matched_excerpts),
                first_excerpt
            ])

    print(f"✅ Exported to CSV: {filepath}")


def export_to_markdown_report(
    results: Dict[str, ProvenanceResult],
    doc_index: DocumentIndex,
    filepath: Path
) -> None:
    """Export multi-category analysis to formatted markdown report"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# Provenance Extraction Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        total_evidence = sum(len(r.evidence) for r in results.values())
        f.write(f"- Categories Analyzed: {len(results)}\n")
        f.write(f"- Total Evidence Found: {total_evidence}\n")
        f.write(f"- Document Nodes Scanned: {list(results.values())[0].total_nodes_scanned}\n\n")

        # Category Breakdown
        f.write("## Category Analysis\n\n")
        for category, result in results.items():
            f.write(f"### {category}\n\n")
            f.write(f"**Topic:** {result.topic}\n\n")
            f.write(f"**Evidence Count:** {len(result.evidence)}\n\n")

            if result.summary:
                f.write(f"**Summary:** {result.summary}\n\n")

            if result.evidence:
                f.write("**Key Evidence:**\n\n")
                for i, match in enumerate(result.evidence[:5], 1):
                    f.write(f"{i}. **[{match.node.node_id}] {match.node.title}** "
                           f"(Score: {match.relevance_score:.2f})\n")
                    f.write(f"   - {match.match_reason}\n")

                    if match.matched_excerpts:
                        f.write(f"   - Excerpt: \"{match.matched_excerpts[0][:200]}...\"\n")
                    f.write("\n")

            f.write("\n")

    print(f"✅ Exported to Markdown: {filepath}")


def create_progress_callback() -> ProgressCallback:
    """Create a progress callback that prints updates"""
    start_time = time.time()

    def callback(update: ProgressUpdate):
        elapsed = time.time() - start_time
        progress_pct = (
            update.current_step / update.total_steps * 100
            if update.total_steps > 0 else 0
        )
        print(f"  [{progress_pct:5.1f}%] {update.step_name} - {update.message} "
              f"({elapsed:.1f}s)")

    return callback


# ============================================================================
# Example 1: Basic Single-Topic Deep Extraction
# ============================================================================

async def example1_basic_extraction(doc_index: DocumentIndex, llm_client) -> None:
    """Demonstrate basic provenance extraction with default configuration"""
    print_section_header("EXAMPLE 1: Basic Single-Topic Deep Extraction")

    print("\n💡 This example shows the fundamental provenance extraction process:")
    print("   - Scans EVERY node in the document (100% coverage)")
    print("   - Returns ALL relevant evidence, not just enough to answer a question")
    print("   - Useful for compliance audits and comprehensive analysis")

    # Create extractor with default configuration
    extractor = ProvenanceExtractor(doc_index, llm_client=llm_client)

    # Extract evidence for cybersecurity
    topic = "cybersecurity risks and data protection"
    print(f"\n🔍 Extracting evidence for: '{topic}'")
    print("   Using default config (threshold=0.6, generate_summary=True)")

    start_time = time.time()
    config = ProvenanceConfig()  # Default configuration
    result = await extractor.extract_all(topic, config)
    elapsed = time.time() - start_time

    # Display results
    display_evidence_summary(result, show_excerpts=True, max_evidence=8)

    print(f"\n⏱️  Extraction completed in {elapsed:.1f} seconds")
    print(f"\n✅ Key Takeaway: Provenance extraction scans every node exhaustively")
    print(f"   and returns ALL relevant evidence ({len(result.evidence)} sections found)")


# ============================================================================
# Example 2: Multi-Category Risk Analysis
# ============================================================================

async def example2_multi_category_analysis(doc_index: DocumentIndex, llm_client) -> None:
    """Demonstrate concurrent extraction across multiple risk categories"""
    print_section_header("EXAMPLE 2: Multi-Category Risk Analysis")

    print("\n💡 This example demonstrates parallel extraction across multiple topics:")
    print("   - Extracts evidence for multiple categories concurrently")
    print("   - Identifies overlap between categories")
    print("   - Reveals interconnected risks")

    # Define risk categories
    risk_categories = {
        "Climate Risks": "climate change, environmental impact, carbon emissions, renewable energy, sustainability",
        "Cyber Risks": "cybersecurity threats, data breaches, privacy, hacking, information security",
        "Regulatory Risks": "compliance, regulations, GDPR, legal requirements, fines, enforcement",
        "Supply Chain Risks": "supply chain disruption, vendor dependencies, logistics, sourcing",
        "Financial Risks": "credit risk, liquidity, market volatility, foreign exchange, economic conditions",
    }

    print(f"\n🔍 Analyzing {len(risk_categories)} risk categories concurrently:")
    for category, topic in risk_categories.items():
        print(f"   - {category}")

    # Create extractor
    extractor = ProvenanceExtractor(doc_index, llm_client=llm_client)

    # Extract evidence for all categories
    start_time = time.time()
    config = ProvenanceConfig(
        relevance_threshold=0.6,
        generate_summary=True,
        parallel_workers=3
    )
    results = await extractor.extract_by_category(risk_categories, config)
    elapsed = time.time() - start_time

    # Display comparison
    display_category_comparison(results)

    # Analyze overlap
    print("\n🔗 Category Overlap Analysis:")
    node_to_categories = defaultdict(list)
    for category, result in results.items():
        for match in result.evidence:
            node_to_categories[match.node.node_id].append(category)

    multi_category_nodes = {
        node_id: categories for node_id, categories in node_to_categories.items()
        if len(categories) > 1
    }

    if multi_category_nodes:
        print(f"\n   Found {len(multi_category_nodes)} sections relevant to multiple categories:")
        for node_id, categories in sorted(
            multi_category_nodes.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:5]:
            node = doc_index.find_node(node_id)
            print(f"\n   [{node_id}] {node.title}")
            print(f"   Categories ({len(categories)}): {', '.join(categories)}")

    print(f"\n⏱️  Multi-category extraction completed in {elapsed:.1f} seconds")
    print(f"\n✅ Key Takeaway: Concurrent extraction is efficient for multiple topics;")
    print(f"   evidence overlap reveals interconnected risks")


# ============================================================================
# Example 3: Evidence Scoring and Threshold Tuning
# ============================================================================

async def example3_scoring_and_thresholds(doc_index: DocumentIndex, llm_client) -> None:
    """Demonstrate the impact of different relevance thresholds"""
    print_section_header("EXAMPLE 3: Evidence Scoring and Threshold Tuning")

    print("\n💡 This example shows how threshold affects precision vs recall:")
    print("   - Lower threshold = more recall (finds more evidence)")
    print("   - Higher threshold = more precision (only high-quality evidence)")
    print("   - Default 0.6 balances comprehensiveness and relevance")

    topic = "artificial intelligence and machine learning"
    print(f"\n🔍 Testing topic: '{topic}'")
    print("   with thresholds: 0.5, 0.6, 0.7, 0.8, 0.9")

    extractor = ProvenanceExtractor(doc_index, llm_client=llm_client)
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

    print("\n📊 Threshold Comparison:")
    print("-" * 70)
    print(f"{'Threshold':<12} {'Results':<12} {'Avg Score':<12} {'Range':<20}")
    print("-" * 70)

    threshold_results = {}
    for threshold in thresholds:
        config = ProvenanceConfig(
            relevance_threshold=threshold,
            generate_summary=False,  # Skip summary for speed
            extract_excerpts=False   # Skip excerpts for speed
        )
        result = await extractor.extract_all(topic, config)
        threshold_results[threshold] = result

        if result.evidence:
            scores = [m.relevance_score for m in result.evidence]
            avg_score = sum(scores) / len(scores)
            score_range = f"{min(scores):.2f} - {max(scores):.2f}"
        else:
            avg_score = 0.0
            score_range = "N/A"

        print(f"{threshold:<12} {len(result.evidence):<12} {avg_score:<12.2f} {score_range:<20}")

    # Show sample evidence at boundaries
    print("\n🔍 Sample Evidence at Threshold Boundaries:")

    for threshold in [0.6, 0.8]:
        result = threshold_results[threshold]
        if result.evidence:
            print(f"\n   Threshold {threshold} - Lowest scoring evidence:")
            lowest = min(result.evidence, key=lambda m: m.relevance_score)
            print(f"   [{lowest.node.node_id}] {lowest.node.title}")
            print(f"   Score: {format_relevance_score(lowest.relevance_score)}")
            print(f"   Reason: {lowest.match_reason}")

    print("\n💡 Recommendations:")
    print("   • Compliance/Audit: Use 0.5-0.6 for comprehensive coverage")
    print("   • Executive Summary: Use 0.8-0.9 for only critical sections")
    print("   • Research: Use 0.6-0.7 for balanced quality and coverage")

    print(f"\n✅ Key Takeaway: Adjust threshold based on your use case to optimize")
    print(f"   the precision-recall trade-off")


# ============================================================================
# Example 4: Summary Generation
# ============================================================================

async def example4_summary_generation(doc_index: DocumentIndex, llm_client) -> None:
    """Demonstrate summary generation from extracted evidence"""
    print_section_header("EXAMPLE 4: Summary Generation")

    print("\n💡 This example compares evidence with and without summaries:")
    print("   - Summaries synthesize findings from all evidence")
    print("   - Useful for executive reports and quick insights")
    print("   - Adds one additional LLM call per extraction")

    topic = "revenue and financial performance"
    print(f"\n🔍 Extracting evidence for: '{topic}'")

    extractor = ProvenanceExtractor(doc_index, llm_client=llm_client)

    # Extract WITHOUT summary
    print("\n📊 Without Summary:")
    config_no_summary = ProvenanceConfig(
        generate_summary=False,
        max_excerpts_per_node=2
    )
    start_time = time.time()
    result_no_summary = await extractor.extract_all(topic, config_no_summary)
    time_no_summary = time.time() - start_time

    print(f"   Evidence found: {len(result_no_summary.evidence)} sections")
    print(f"   Time: {time_no_summary:.1f}s")

    # Extract WITH summary
    print("\n📊 With Summary:")
    config_with_summary = ProvenanceConfig(
        generate_summary=True,
        max_excerpts_per_node=2
    )
    start_time = time.time()
    result_with_summary = await extractor.extract_all(topic, config_with_summary)
    time_with_summary = time.time() - start_time

    print(f"   Evidence found: {len(result_with_summary.evidence)} sections")
    print(f"   Time: {time_with_summary:.1f}s")
    print(f"\n   📝 Generated Summary:")
    print(f"   {result_with_summary.summary}")

    # Show cost-benefit
    print(f"\n⏱️  Time Comparison:")
    print(f"   Without summary: {time_no_summary:.1f}s")
    print(f"   With summary: {time_with_summary:.1f}s")
    print(f"   Overhead: {time_with_summary - time_no_summary:.1f}s "
          f"({((time_with_summary - time_no_summary) / time_no_summary * 100):.0f}% increase)")

    print(f"\n✅ Key Takeaway: Summaries add modest overhead but provide valuable")
    print(f"   synthesis of findings for executive reporting")


# ============================================================================
# Example 5: Progress Tracking for Large Documents
# ============================================================================

async def example5_progress_tracking(doc_index: DocumentIndex, llm_client) -> None:
    """Demonstrate progress tracking for long-running extractions"""
    print_section_header("EXAMPLE 5: Progress Tracking for Large Documents")

    print("\n💡 This example shows real-time progress tracking:")
    print("   - Provides visibility into long-running operations")
    print("   - Shows batch processing progress")
    print("   - Allows estimation of remaining time")

    topic = "product development and innovation"
    print(f"\n🔍 Extracting with progress tracking: '{topic}'")

    extractor = ProvenanceExtractor(doc_index, llm_client=llm_client)

    # Create progress callback
    progress_callback = create_progress_callback()

    # Extract with progress
    config = ProvenanceConfig(
        batch_size=15,
        parallel_workers=5,
        extract_excerpts=True
    )

    print("\n⏳ Progress Updates:")
    result = await extractor.extract_with_progress(
        topic,
        config,
        progress_callback=progress_callback
    )

    # Display results
    print("\n" + "-" * 70)
    print(f"📚 Evidence Found: {len(result.evidence)} relevant sections")
    print(f"📊 Scan Coverage: {result.scan_coverage*100:.0f}%")

    if result.evidence:
        print(f"\n🔝 Top 3 Evidence Pieces:")
        for i, match in enumerate(result.evidence[:3], 1):
            print(f"\n   {i}. [{match.node.node_id}] {match.node.title}")
            print(f"      Score: {format_relevance_score(match.relevance_score)}")

    print(f"\n✅ Key Takeaway: Progress tracking is essential for large documents")
    print(f"   and provides users with visibility during long operations")


# ============================================================================
# Example 6: Streaming Extraction
# ============================================================================

async def example6_streaming_extraction(doc_index: DocumentIndex, llm_client) -> None:
    """Demonstrate real-time streaming extraction"""
    print_section_header("EXAMPLE 6: Streaming Extraction")

    print("\n💡 This example demonstrates streaming results as they're found:")
    print("   - Results arrive in real-time as nodes are processed")
    print("   - Enables immediate analysis without waiting for completion")
    print("   - Ideal for interactive applications and dashboards")

    topic = "supply chain and operations"
    print(f"\n🔍 Streaming extraction for: '{topic}'")

    extractor = ProvenanceExtractor(doc_index, llm_client=llm_client)

    # Start streaming extraction
    config = ProvenanceConfig(
        batch_size=10,
        extract_excerpts=True,
        max_excerpts_per_node=2
    )

    print("\n🌊 Streaming Evidence Discovery:")
    print("-" * 70)

    result = await extractor.extract_stream(topic, config)

    # Process results as they stream in
    evidence_count = 0
    start_time = time.time()

    async for match in result.evidence_stream:
        evidence_count += 1
        elapsed = time.time() - start_time

        print(f"\n   ✓ [{match.node.node_id}] {match.node.title}")
        print(f"     Score: {format_relevance_score(match.relevance_score)}")
        print(f"     Time: {elapsed:.1f}s | Count: {evidence_count}")

        if match.matched_excerpts:
            preview = match.matched_excerpts[0][:120] + "..."
            print(f"     Excerpt: \"{preview}\"")

    print("\n" + "-" * 70)
    print(f"🏁 Streaming completed: {evidence_count} evidence pieces found")

    print(f"\n✅ Key Takeaway: Streaming enables real-time user feedback and allows")
    print(f"   early analysis before the full extraction completes")


# ============================================================================
# Example 7: Export Formats and Reporting
# ============================================================================

async def example7_export_formats(doc_index: DocumentIndex, llm_client) -> None:
    """Demonstrate various export formats for provenance results"""
    print_section_header("EXAMPLE 7: Export Formats and Reporting")

    print("\n💡 This example shows multiple export formats:")
    print("   - JSON for programmatic processing")
    print("   - CSV for spreadsheet analysis")
    print("   - Markdown for human-readable reports")

    # Perform multi-category extraction
    categories = {
        "Technology": "technology, innovation, R&D, digital transformation",
        "Competition": "competitive landscape, market position, competitors",
        "Regulatory": "regulations, compliance, legal requirements",
    }

    print(f"\n🔍 Extracting evidence for export demonstration:")
    for category in categories:
        print(f"   - {category}")

    extractor = ProvenanceExtractor(doc_index, llm_client=llm_client)
    config = ProvenanceConfig(
        relevance_threshold=0.7,
        generate_summary=True,
        max_excerpts_per_node=3
    )

    results = await extractor.extract_by_category(categories, config)

    # Create exports directory
    export_dir = Path(__file__).parent.parent / "exports"
    export_dir.mkdir(exist_ok=True)

    print(f"\n📁 Exporting to: {export_dir}")

    # Export to JSON (first category as example)
    json_path = export_dir / "provenance_results.json"
    export_to_json(results["Technology"], json_path)

    # Export to CSV
    csv_path = export_dir / "provenance_evidence.csv"
    export_to_csv(results["Technology"], doc_index, csv_path)

    # Export to Markdown
    md_path = export_dir / "provenance_report.md"
    export_to_markdown_report(results, doc_index, md_path)

    # Show file sizes
    print(f"\n📊 Generated Files:")
    for filepath in [json_path, csv_path, md_path]:
        size_kb = filepath.stat().st_size / 1024
        print(f"   {filepath.name}: {size_kb:.1f} KB")

    # Show sample from each format
    print(f"\n📄 Sample JSON Output:")
    with open(json_path, 'r') as f:
        sample_json = json.load(f)
        print(f"   Topic: {sample_json['topic']}")
        print(f"   Evidence Count: {len(sample_json['evidence'])}")
        print(f"   Total Nodes Scanned: {sample_json['total_nodes_scanned']}")

    print(f"\n📄 Sample CSV Output (first 3 rows):")
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                print(f"   Columns: {', '.join(row)}")
            elif i <= 3:
                print(f"   Row {i}: {row[0]}, {row[1]}, Score: {row[2]}")

    print(f"\n💡 Use Cases:")
    print(f"   • JSON: Integration with analysis pipelines, APIs, databases")
    print(f"   • CSV: Excel analysis, filtering, pivot tables")
    print(f"   • Markdown: Reports, documentation, sharing with stakeholders")

    print(f"\n✅ Key Takeaway: Multiple export formats support different audiences")
    print(f"   and use cases for provenance data")


# ============================================================================
# Document Preparation
# ============================================================================

async def create_indexed_document(llm_client) -> DocumentIndex:
    """Create indexed document for provenance examples"""
    print("\n🔧 Preparing document index...")

    # Load Apple 10-K document
    doc_path = Path(__file__).parent.parent / "docs" / "AAPL_10-K_2024-11-01.txt"

    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    with open(doc_path, 'r', encoding='utf-8') as f:
        document_text = f.read()

    print(f"📄 Loaded document: {doc_path.name}")
    print(f"   Size: {len(document_text):,} characters")

    # Create indexer configuration
    config = IndexerConfig(
        llm_config=llm_client.config,
        generate_summaries=True,
        extract_metadata=True,
        resolve_cross_refs=True,
    )

    # Index the document
    indexer = DocumentIndexer(config)
    doc_index = await indexer.index(
        text=document_text,
        doc_name="AAPL_10-K_2024"
    )

    print(f"✓ Document indexed: {doc_index.get_node_count()} nodes")
    print(f"✓ Document type: {doc_index.doc_type.value}")

    return doc_index


# ============================================================================
# Main Orchestrator
# ============================================================================

async def main():
    """Run all provenance extraction examples"""
    print("\n" + "#" * 70)
    print("Provenance Extraction Patterns")
    print("Exhaustive Evidence Discovery with DocumentIndex")
    print("#" * 70)

    # Load environment and create LLM client
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)

    provider = os.getenv("DOCUMENTINDEX_LLM_PROVIDER", "bedrock").lower()

    if provider == "bedrock":
        llm_client = create_bedrock_client(
            model=os.getenv(
                "DOCUMENTINDEX_LLM_MODEL",
                "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
            ),
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

    print(f"\n🤖 Using LLM: {provider} / {llm_client.config.model}")

    # Create indexed document
    doc_index = await create_indexed_document(llm_client)

    # Run examples sequentially
    await example1_basic_extraction(doc_index, llm_client)
    await example2_multi_category_analysis(doc_index, llm_client)
    await example3_scoring_and_thresholds(doc_index, llm_client)
    await example4_summary_generation(doc_index, llm_client)
    await example5_progress_tracking(doc_index, llm_client)
    await example6_streaming_extraction(doc_index, llm_client)
    await example7_export_formats(doc_index, llm_client)

    print("\n" + "=" * 70)
    print("✅ All provenance extraction examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
