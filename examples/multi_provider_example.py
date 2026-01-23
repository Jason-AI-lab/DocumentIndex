"""
Multi-Provider LLM Example: Using different LLM providers.

This example demonstrates:
1. Using OpenAI
2. Using Anthropic (Claude)
3. Using AWS Bedrock
4. Using Azure OpenAI
5. Using local models (Ollama)
"""

import asyncio
from documentindex import (
    LLMConfig,
    LLMClient,
    create_openai_client,
    create_anthropic_client,
    create_bedrock_client,
    create_azure_client,
    create_ollama_client,
    DocumentIndexer,
    IndexerConfig,
)


SAMPLE_TEXT = """
Company Overview

TechCorp is a leading technology company specializing in cloud computing 
and artificial intelligence solutions. Founded in 2010, we have grown to 
serve Fortune 500 companies worldwide.

Financial Highlights

2024 Revenue: $5 billion
2024 Net Income: $800 million
Employees: 15,000
"""


async def openai_example():
    """Using OpenAI models"""
    print("=" * 60)
    print("EXAMPLE 1: OpenAI")
    print("=" * 60)
    
    # Method 1: Using factory function
    client = create_openai_client(
        model="gpt-4o",
        # api_key="your-key",  # Or use OPENAI_API_KEY env var
    )
    
    # Method 2: Using LLMConfig directly
    config = LLMConfig(
        model="gpt-4o",  # or "openai/gpt-4-turbo"
        temperature=0.0,
        max_tokens=1000,
    )
    client = LLMClient(config)
    
    print("Client configured for OpenAI GPT-4o")
    
    # Use with indexer
    indexer_config = IndexerConfig(llm_config=config)
    indexer = DocumentIndexer(indexer_config)
    
    print("Indexer ready with OpenAI backend")


async def anthropic_example():
    """Using Anthropic Claude"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Anthropic Claude")
    print("=" * 60)
    
    # Method 1: Using factory function
    client = create_anthropic_client(
        model="claude-sonnet-4-20250514",
        # api_key="your-key",  # Or use ANTHROPIC_API_KEY env var
    )
    
    # Method 2: Using LLMConfig
    config = LLMConfig(
        model="anthropic/claude-sonnet-4-20250514",
        temperature=0.0,
        max_tokens=4096,
    )
    
    print("Client configured for Anthropic Claude Sonnet")
    
    # Use with indexer
    indexer_config = IndexerConfig(llm_config=config)
    indexer = DocumentIndexer(indexer_config)
    
    print("Indexer ready with Anthropic backend")


async def bedrock_example():
    """Using AWS Bedrock"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: AWS Bedrock")
    print("=" * 60)
    
    # Method 1: Using factory function
    client = create_bedrock_client(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        region="us-east-1",
    )
    
    # Method 2: Using LLMConfig
    config = LLMConfig(
        model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        provider_config={
            "aws_region_name": "us-east-1",
        },
    )
    
    print("Client configured for AWS Bedrock")
    print("Note: Requires AWS credentials (env vars, ~/.aws/credentials, or IAM role)")
    
    # Available Bedrock models:
    print("\nAvailable Bedrock models:")
    print("  - bedrock/anthropic.claude-3-sonnet-20240229-v1:0")
    print("  - bedrock/anthropic.claude-3-haiku-20240307-v1:0")
    print("  - bedrock/amazon.titan-text-express-v1")
    print("  - bedrock/meta.llama2-70b-chat-v1")


async def azure_example():
    """Using Azure OpenAI"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Azure OpenAI")
    print("=" * 60)
    
    # Method 1: Using factory function
    client = create_azure_client(
        deployment_name="gpt-4-deployment",
        api_base="https://your-resource.openai.azure.com",
        api_version="2024-02-15-preview",
        # api_key="your-key",  # Or use AZURE_API_KEY env var
    )
    
    # Method 2: Using LLMConfig
    config = LLMConfig(
        model="azure/gpt-4-deployment",
        provider_config={
            "api_base": "https://your-resource.openai.azure.com",
            "api_version": "2024-02-15-preview",
        },
        # api_key="your-key",
    )
    
    print("Client configured for Azure OpenAI")
    print("\nConfiguration required:")
    print("  - deployment_name: Your Azure deployment name")
    print("  - api_base: Your Azure OpenAI endpoint")
    print("  - api_version: API version (e.g., 2024-02-15-preview)")
    print("  - api_key: Azure API key (or set AZURE_API_KEY)")


async def ollama_example():
    """Using local Ollama models"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Ollama (Local)")
    print("=" * 60)
    
    # Method 1: Using factory function
    client = create_ollama_client(
        model="llama2",
        base_url="http://localhost:11434",
    )
    
    # Method 2: Using LLMConfig
    config = LLMConfig(
        model="ollama/llama2",
        provider_config={
            "api_base": "http://localhost:11434",
        },
    )
    
    print("Client configured for Ollama")
    print("\nPrerequisites:")
    print("  1. Install Ollama: https://ollama.ai")
    print("  2. Pull a model: ollama pull llama2")
    print("  3. Start Ollama server (usually automatic)")
    
    print("\nAvailable models (after pulling):")
    print("  - ollama/llama2")
    print("  - ollama/mistral")
    print("  - ollama/codellama")
    print("  - ollama/phi")


async def complete_workflow_example():
    """Complete workflow with a specific provider"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Complete Workflow")
    print("=" * 60)
    
    # Configure for OpenAI (change to your preferred provider)
    llm_config = LLMConfig(
        model="gpt-4o",
        temperature=0.0,
        max_tokens=4096,
        max_retries=3,
        retry_delay=1.0,
        timeout=120.0,
    )
    
    # Create indexer
    indexer_config = IndexerConfig(
        llm_config=llm_config,
        generate_summaries=True,
        extract_metadata=True,
    )
    indexer = DocumentIndexer(indexer_config)
    
    print("Indexing document...")
    doc_index = await indexer.index(
        text=SAMPLE_TEXT,
        doc_name="company_overview",
    )
    
    print(f"Document indexed: {doc_index.get_node_count()} nodes")
    
    # Query the document
    from documentindex import AgenticQA, AgenticQAConfig
    
    qa = AgenticQA(doc_index, llm_config=llm_config)
    
    question = "What was TechCorp's revenue in 2024?"
    print(f"\nQuestion: {question}")
    
    result = await qa.answer(question)
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.2f}")


async def main():
    """Run provider examples"""
    print("\n" + "#" * 60)
    print("DocumentIndex Multi-Provider Examples")
    print("#" * 60)
    
    # Show configurations (without actually calling APIs)
    await openai_example()
    await anthropic_example()
    await bedrock_example()
    await azure_example()
    await ollama_example()
    
    # Uncomment to run actual workflow (requires API keys)
    # await complete_workflow_example()
    
    print("\n" + "=" * 60)
    print("Multi-provider examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
