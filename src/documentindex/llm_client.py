"""
LLM Client supporting multiple providers with streaming.

Supported providers via litellm:
- OpenAI (openai/gpt-4o, openai/gpt-4-turbo)
- Anthropic (anthropic/claude-sonnet-4-20250514, anthropic/claude-3-haiku)
- AWS Bedrock (bedrock/anthropic.claude-3-sonnet, bedrock/amazon.titan-text)
- Azure OpenAI (azure/gpt-4, azure/gpt-35-turbo)
- Local models via Ollama (ollama/llama2, ollama/mistral)
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Literal, AsyncIterator
import asyncio
import json
import re
import logging

from .streaming import StreamChunk, ProgressCallback, ProgressUpdate, OperationType

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM client"""
    
    # Model specification (litellm format)
    # Examples:
    #   - "gpt-4o" or "openai/gpt-4o"
    #   - "anthropic/claude-sonnet-4-20250514"
    #   - "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
    #   - "azure/gpt-4-deployment-name"
    #   - "ollama/llama2"
    model: str = "gpt-4o"
    
    # Generation parameters
    temperature: float = 0.0
    max_tokens: int = 4096
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 120.0
    
    # Provider-specific configuration
    # For Azure: {"api_base": "https://xxx.openai.azure.com", "api_version": "2024-02-15"}
    # For Bedrock: {"aws_region_name": "us-east-1"}
    provider_config: dict[str, Any] = field(default_factory=dict)
    
    # API keys (if not using environment variables)
    api_key: Optional[str] = None


class LLMClient:
    """
    Async LLM client supporting multiple providers via litellm.
    
    Supported providers and model formats:
    
    OpenAI:
        model="gpt-4o"
        model="openai/gpt-4-turbo"
        
    Anthropic:
        model="anthropic/claude-sonnet-4-20250514"
        model="anthropic/claude-3-haiku-20240307"
        
    AWS Bedrock:
        model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
        model="bedrock/amazon.titan-text-express-v1"
        model="bedrock/meta.llama2-70b-chat-v1"
        provider_config={"aws_region_name": "us-east-1"}
        
    Azure OpenAI:
        model="azure/your-deployment-name"
        provider_config={
            "api_base": "https://your-resource.openai.azure.com",
            "api_version": "2024-02-15-preview"
        }
        
    Ollama (local):
        model="ollama/llama2"
        model="ollama/mistral"
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._litellm = None
    
    async def _get_litellm(self):
        """Lazy import litellm to avoid import errors if not installed"""
        if self._litellm is None:
            try:
                import litellm
                
                # Disable litellm's verbose logging
                litellm.suppress_debug_info = True
                
                # Configure provider-specific settings
                if self.config.provider_config:
                    for key, value in self.config.provider_config.items():
                        if hasattr(litellm, key):
                            setattr(litellm, key, value)
                
                # Set API key if provided
                if self.config.api_key:
                    litellm.api_key = self.config.api_key
                
                self._litellm = litellm
            except ImportError:
                raise ImportError(
                    "litellm is required for LLM operations. "
                    "Install it with: pip install litellm"
                )
        return self._litellm
    
    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Literal["text", "json"] = "text",
    ) -> list[dict[str, str]]:
        """Build messages list for LLM call"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        user_content = prompt
        if response_format == "json":
            user_content += "\n\nRespond with valid JSON only, no markdown code blocks."
        
        messages.append({"role": "user", "content": user_content})
        return messages
    
    def _build_kwargs(self, messages: list[dict], stream: bool = False) -> dict:
        """Build kwargs for litellm call"""
        kwargs = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "timeout": self.config.timeout,
            "stream": stream,
        }
        
        # Add provider-specific config (excluding already set attributes)
        for key, value in self.config.provider_config.items():
            if key not in kwargs:
                kwargs[key] = value
        
        return kwargs
    
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Literal["text", "json"] = "text",
    ) -> str:
        """
        Get completion from LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            response_format: "text" or "json" (adds JSON instruction if "json")
        
        Returns:
            LLM response text
        """
        litellm = await self._get_litellm()
        
        messages = self._build_messages(prompt, system_prompt, response_format)
        kwargs = self._build_kwargs(messages, stream=False)
        
        # Retry loop
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = await litellm.acompletion(**kwargs)
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        raise RuntimeError(f"LLM call failed after {self.config.max_retries} retries: {last_error}")
    
    async def complete_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """Get JSON completion from LLM with automatic parsing"""
        response = await self.complete(prompt, system_prompt, response_format="json")
        return self._extract_json(response)
    
    async def complete_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Literal["text", "json"] = "text",
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream completion from LLM.
        
        Yields StreamChunk objects as tokens arrive.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            response_format: "text" or "json"
        
        Yields:
            StreamChunk with incremental content
        
        Example:
            async for chunk in client.complete_stream("Explain quantum computing"):
                print(chunk.content, end="", flush=True)
                if chunk.is_complete:
                    print(f"\\n\\nTotal tokens: {chunk.token_count}")
        """
        litellm = await self._get_litellm()
        
        messages = self._build_messages(prompt, system_prompt, response_format)
        kwargs = self._build_kwargs(messages, stream=True)
        
        accumulated = ""
        token_count = 0
        
        try:
            response = await litellm.acompletion(**kwargs)
            
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    accumulated += content
                    token_count += 1  # Approximate token count
                    
                    yield StreamChunk(
                        content=content,
                        is_complete=False,
                        token_count=token_count,
                        accumulated_content=accumulated,
                    )
            
            # Final chunk
            yield StreamChunk(
                content="",
                is_complete=True,
                token_count=token_count,
                accumulated_content=accumulated,
            )
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            # Yield error as final chunk
            yield StreamChunk(
                content="",
                is_complete=True,
                token_count=token_count,
                accumulated_content=accumulated,
                error=str(e),
            )
    
    async def complete_stream_collect(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Literal["text", "json"] = "text",
    ) -> str:
        """Stream completion and collect full result"""
        result = ""
        async for chunk in self.complete_stream(prompt, system_prompt, response_format):
            result = chunk.accumulated_content
            if chunk.error:
                raise RuntimeError(chunk.error)
        return result
    
    async def complete_chunked(
        self,
        prompt_parts: list[str],
        system_prompt: Optional[str] = None,
        combine_strategy: Literal["concat", "summarize", "last"] = "concat",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> str:
        """
        Process large input by chunking into multiple LLM calls.
        
        Useful when input exceeds context window.
        
        Args:
            prompt_parts: List of prompt parts to process
            system_prompt: System prompt for all calls
            combine_strategy: How to combine results
                - "concat": Concatenate all results
                - "summarize": Summarize combined results
                - "last": Use only the last result (for iterative refinement)
            progress_callback: Called after each chunk
        
        Returns:
            Combined result string
        """
        results = []
        total = len(prompt_parts)
        
        for i, part in enumerate(prompt_parts):
            result = await self.complete(part, system_prompt)
            results.append(result)
            
            if progress_callback:
                progress_callback(ProgressUpdate(
                    operation=OperationType.INDEXING,
                    current_step=i + 1,
                    total_steps=total,
                    step_name=f"Processing chunk {i + 1}/{total}",
                    message=f"Processed {len(part)} characters",
                ))
        
        if combine_strategy == "concat":
            return "\n\n".join(results)
        elif combine_strategy == "last":
            return results[-1] if results else ""
        elif combine_strategy == "summarize":
            if not results:
                return ""
            combined = "\n\n".join(results)
            summary_prompt = f"Summarize and consolidate these results:\n\n{combined}"
            return await self.complete(summary_prompt, system_prompt)
        
        return results[-1] if results else ""
    
    @staticmethod
    def _extract_json(text: str) -> dict:
        """Extract and parse JSON from LLM response"""
        if not text:
            return {}
        
        # Try to find JSON in code blocks first
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if json_match:
            text = json_match.group(1)
        
        text = text.strip()
        
        # Clean common issues
        text = re.sub(r',\s*}', '}', text)  # Trailing commas in objects
        text = re.sub(r',\s*]', ']', text)  # Trailing commas in arrays
        
        # Handle Python-style values
        text = text.replace('None', 'null')
        text = text.replace('True', 'true')
        text = text.replace('False', 'false')
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object or array
            for start, end in [('{', '}'), ('[', ']')]:
                start_idx = text.find(start)
                end_idx = text.rfind(end)
                if start_idx != -1 and end_idx > start_idx:
                    try:
                        return json.loads(text[start_idx:end_idx + 1])
                    except json.JSONDecodeError:
                        continue
            
            logger.warning(f"Failed to parse JSON from response: {text[:200]}...")
            return {}


# ============================================================================
# Convenience factory functions
# ============================================================================

def create_openai_client(
    model: str = "gpt-4o",
    api_key: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """Create client for OpenAI"""
    model_str = model if "/" in model else f"openai/{model}"
    return LLMClient(LLMConfig(
        model=model_str,
        api_key=api_key,
        **kwargs,
    ))


def create_anthropic_client(
    model: str = "claude-sonnet-4-20250514",
    api_key: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """Create client for Anthropic"""
    model_str = model if model.startswith("anthropic/") else f"anthropic/{model}"
    return LLMClient(LLMConfig(
        model=model_str,
        api_key=api_key,
        **kwargs,
    ))


def create_bedrock_client(
    model: str = "anthropic.claude-3-sonnet-20240229-v1:0",
    region: str = "us-east-1",
    **kwargs,
) -> LLMClient:
    """
    Create client for AWS Bedrock.
    
    Requires AWS credentials configured via:
    - Environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    - AWS credentials file (~/.aws/credentials)
    - IAM role (when running on AWS)
    
    Common Bedrock models:
    - anthropic.claude-3-sonnet-20240229-v1:0
    - anthropic.claude-3-haiku-20240307-v1:0
    - amazon.titan-text-express-v1
    - meta.llama2-70b-chat-v1
    """
    model_str = model if model.startswith("bedrock/") else f"bedrock/{model}"
    return LLMClient(LLMConfig(
        model=model_str,
        provider_config={"aws_region_name": region},
        **kwargs,
    ))


def create_azure_client(
    deployment_name: str,
    api_base: str,
    api_version: str = "2024-02-15-preview",
    api_key: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """
    Create client for Azure OpenAI.
    
    Args:
        deployment_name: Your Azure deployment name
        api_base: Your Azure endpoint (https://xxx.openai.azure.com)
        api_version: API version
        api_key: Azure API key (or use AZURE_API_KEY env var)
    """
    return LLMClient(LLMConfig(
        model=f"azure/{deployment_name}",
        api_key=api_key,
        provider_config={
            "api_base": api_base,
            "api_version": api_version,
        },
        **kwargs,
    ))


def create_ollama_client(
    model: str = "llama2",
    base_url: str = "http://localhost:11434",
    **kwargs,
) -> LLMClient:
    """Create client for local Ollama models"""
    model_str = model if model.startswith("ollama/") else f"ollama/{model}"
    return LLMClient(LLMConfig(
        model=model_str,
        provider_config={"api_base": base_url},
        **kwargs,
    ))
