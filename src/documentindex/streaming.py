"""
Streaming support for DocumentIndex.

Provides:
- Streaming LLM responses
- Progress tracking for long operations
- Chunked processing for large documents
- Async generators for incremental results
"""

from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional, Any, Awaitable, TYPE_CHECKING
from enum import Enum
import asyncio
from datetime import datetime

if TYPE_CHECKING:
    from .models import NodeMatch


# ============================================================================
# Progress Tracking
# ============================================================================

class OperationType(Enum):
    """Types of long-running operations"""
    INDEXING = "indexing"
    SEARCHING = "searching"
    QA = "question_answering"
    PROVENANCE = "provenance_extraction"
    METADATA = "metadata_extraction"
    CROSS_REF = "cross_reference_resolution"


@dataclass
class ProgressUpdate:
    """Progress update for long-running operations"""
    operation: OperationType
    current_step: int
    total_steps: int
    step_name: str
    message: str = ""
    started_at: datetime = field(default_factory=datetime.now)
    data: Any = None  # Optional data payload (e.g., final result)
    
    @property
    def progress_pct(self) -> float:
        """Progress as percentage (0-100)"""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100
    
    @property
    def is_complete(self) -> bool:
        return self.current_step >= self.total_steps
    
    def __repr__(self) -> str:
        return f"ProgressUpdate({self.operation.value}, {self.progress_pct:.1f}%, {self.step_name})"


# Type aliases for progress callbacks
ProgressCallback = Callable[[ProgressUpdate], None]
AsyncProgressCallback = Callable[[ProgressUpdate], Awaitable[None]]


# ============================================================================
# Streaming Chunks
# ============================================================================

@dataclass
class StreamChunk:
    """A chunk of streamed LLM response"""
    content: str
    is_complete: bool = False
    token_count: int = 0
    accumulated_content: str = ""  # Full content so far
    error: Optional[str] = None  # Error message if any
    
    def __repr__(self) -> str:
        if self.is_complete:
            return f"StreamChunk(complete, {self.token_count} tokens)"
        return f"StreamChunk({len(self.content)} chars)"


# ============================================================================
# Streaming Result Containers
# ============================================================================

@dataclass
class StreamingQAResult:
    """Streaming result from agentic QA"""
    question: str
    
    # Streamed answer chunks - async iterator
    answer_stream: AsyncIterator[StreamChunk]
    
    # Available after streaming completes
    confidence: float = 0.0
    citations: list = field(default_factory=list)
    reasoning_trace: list = field(default_factory=list)
    nodes_visited: list = field(default_factory=list)
    
    # For collecting the full answer
    _accumulated_answer: str = ""
    
    async def get_full_answer(self) -> str:
        """Consume stream and return full answer"""
        if self._accumulated_answer:
            return self._accumulated_answer
        
        async for chunk in self.answer_stream:
            if chunk.is_complete:
                self._accumulated_answer = chunk.accumulated_content
                break
        
        return self._accumulated_answer


@dataclass
class StreamingProvenanceResult:
    """Streaming result from provenance extraction"""
    topic: str
    
    # Yields NodeMatch objects as they're found
    evidence_stream: AsyncIterator["NodeMatch"]
    
    # Available after streaming completes
    total_nodes_scanned: int = 0
    scan_coverage: float = 0.0
    
    # For collecting all evidence
    _collected_evidence: list = field(default_factory=list)
    
    async def collect_all(self) -> list["NodeMatch"]:
        """Consume stream and return all evidence"""
        if self._collected_evidence:
            return self._collected_evidence
        
        async for match in self.evidence_stream:
            self._collected_evidence.append(match)
        
        return self._collected_evidence


# ============================================================================
# Progress Reporter Helper
# ============================================================================

class ProgressReporter:
    """Helper class for reporting progress"""
    
    def __init__(
        self,
        operation: OperationType,
        total_steps: int,
        callback: Optional[ProgressCallback] = None,
        async_callback: Optional[AsyncProgressCallback] = None,
    ):
        self.operation = operation
        self.total_steps = total_steps
        self.callback = callback
        self.async_callback = async_callback
        self.current_step = 0
        self.started_at = datetime.now()
    
    def report(self, step_name: str, message: str = "", data: Any = None) -> ProgressUpdate:
        """Report progress (sync)"""
        self.current_step += 1
        update = ProgressUpdate(
            operation=self.operation,
            current_step=self.current_step,
            total_steps=self.total_steps,
            step_name=step_name,
            message=message,
            started_at=self.started_at,
            data=data,
        )
        
        if self.callback:
            self.callback(update)
        
        return update
    
    async def report_async(self, step_name: str, message: str = "", data: Any = None) -> ProgressUpdate:
        """Report progress (async)"""
        update = self.report(step_name, message, data)
        
        if self.async_callback:
            await self.async_callback(update)
        
        return update
    
    def set_total(self, total: int):
        """Update total steps"""
        self.total_steps = total


# ============================================================================
# Async Stream Utilities
# ============================================================================

async def collect_stream(stream: AsyncIterator[StreamChunk]) -> str:
    """Collect all chunks from a stream into a single string"""
    result = ""
    async for chunk in stream:
        result = chunk.accumulated_content
        if chunk.is_complete:
            break
    return result


async def stream_to_list(stream: AsyncIterator[Any]) -> list:
    """Convert async iterator to list"""
    return [item async for item in stream]


class AsyncBuffer:
    """Buffer for accumulating async stream content"""
    
    def __init__(self):
        self.content = ""
        self.chunks: list[str] = []
        self.token_count = 0
        self.is_complete = False
        self.error: Optional[str] = None
    
    def add(self, chunk: StreamChunk):
        """Add a chunk to the buffer"""
        self.chunks.append(chunk.content)
        self.content = chunk.accumulated_content
        self.token_count = chunk.token_count
        self.is_complete = chunk.is_complete
        if chunk.error:
            self.error = chunk.error
    
    def get_content(self) -> str:
        """Get accumulated content"""
        return self.content
