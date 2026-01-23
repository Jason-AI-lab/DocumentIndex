"""
Agentic Question Answering - iterative, intelligent retrieval for QA.

Workflow:
1. Understand the question
2. Examine document structure
3. Select most promising section
4. Read and extract information
5. If sufficient: answer; else: continue searching
6. Follow cross-references as needed
"""

from dataclasses import dataclass, field
from typing import Optional, AsyncIterator
import logging

from .models import DocumentIndex, TreeNode, QAResult, Citation
from .llm_client import LLMClient, LLMConfig
from .cross_ref import CrossReferenceFollower
from .cache import CacheManager
from .streaming import (
    StreamChunk, StreamingQAResult,
    ProgressCallback, ProgressUpdate, OperationType, ProgressReporter,
)

logger = logging.getLogger(__name__)


@dataclass
class AgenticQAConfig:
    """Configuration for agentic QA"""
    max_iterations: int = 5
    max_context_tokens: int = 8000
    follow_cross_refs: bool = True
    generate_citations: bool = True
    confidence_threshold: float = 0.7  # Min confidence to stop searching


@dataclass
class ReasoningStep:
    """A step in the reasoning process"""
    step_num: int
    action: str  # "plan", "read_section", "follow_reference", "answer", "give_up"
    node_id: Optional[str] = None
    reasoning: str = ""
    findings: str = ""
    
    def to_dict(self) -> dict:
        return {
            "step": self.step_num,
            "action": self.action,
            "node_id": self.node_id,
            "reasoning": self.reasoning,
            "findings": self.findings,
        }


class AgenticQA:
    """
    Agentic question answering system.
    
    Uses iterative reasoning to find and synthesize information
    from the document to answer questions.
    """
    
    def __init__(
        self,
        doc_index: DocumentIndex,
        llm_client: Optional[LLMClient] = None,
        llm_config: Optional[LLMConfig] = None,
        cache_manager: Optional[CacheManager] = None,
    ):
        self.doc_index = doc_index
        self.llm = llm_client or LLMClient(llm_config or LLMConfig())
        self.cache = cache_manager
        self.cross_ref_follower = CrossReferenceFollower(doc_index)
    
    async def answer(
        self,
        question: str,
        config: Optional[AgenticQAConfig] = None,
    ) -> QAResult:
        """
        Answer a question using agentic retrieval.
        
        Args:
            question: The question to answer
            config: QA configuration
        
        Returns:
            QAResult with answer, citations, and reasoning trace
        """
        config = config or AgenticQAConfig()
        
        # Initialize state
        reasoning_trace: list[ReasoningStep] = []
        visited_nodes: list[str] = []
        gathered_info: list[tuple[str, str, str]] = []  # (node_id, title, excerpt)
        
        # Step 1: Plan the search
        plan = await self._plan_search(question)
        reasoning_trace.append(ReasoningStep(
            step_num=1,
            action="plan",
            reasoning=plan.get("reasoning", ""),
            findings=f"Search targets: {plan.get('targets', [])}",
        ))
        
        # Iterative search loop
        for iteration in range(config.max_iterations):
            # Get current structure view
            structure_view = self._get_structure_view(visited_nodes)
            
            # Decide next action
            decision = await self._decide_next_action(
                question=question,
                structure=structure_view,
                gathered_info=gathered_info,
                plan=plan,
            )
            
            action = decision.get("action", "give_up")
            
            if action == "answer":
                # We have enough info to answer
                reasoning_trace.append(ReasoningStep(
                    step_num=len(reasoning_trace) + 1,
                    action="answer",
                    reasoning=decision.get("reasoning", "Sufficient information gathered"),
                ))
                break
            
            elif action == "read_section":
                node_id = decision.get("node_id")
                if not node_id or node_id in visited_nodes:
                    continue
                
                visited_nodes.append(node_id)
                
                # Read the section
                node = self.doc_index.find_node(node_id)
                if not node:
                    continue
                
                text = self.doc_index.get_node_text(node_id)
                if not text:
                    continue
                
                # Extract relevant information
                extraction = await self._extract_relevant_info(question, text)
                
                findings = extraction.get("summary", "No relevant information found")
                
                if extraction.get("found_relevant"):
                    gathered_info.append((
                        node_id,
                        node.title,
                        extraction.get("excerpt", ""),
                    ))
                    findings = extraction.get("summary", "Found relevant information")
                
                reasoning_trace.append(ReasoningStep(
                    step_num=len(reasoning_trace) + 1,
                    action="read_section",
                    node_id=node_id,
                    reasoning=decision.get("reasoning", ""),
                    findings=findings,
                ))
                
                # Follow cross-references if enabled and suggested
                if config.follow_cross_refs and extraction.get("follow_refs"):
                    for ref in node.cross_references:
                        if ref.resolved and ref.target_node_id and ref.target_node_id not in visited_nodes:
                            reasoning_trace.append(ReasoningStep(
                                step_num=len(reasoning_trace) + 1,
                                action="follow_reference",
                                node_id=ref.target_node_id,
                                reasoning=f"Following reference to {ref.target_description}",
                            ))
            
            elif action == "give_up":
                reasoning_trace.append(ReasoningStep(
                    step_num=len(reasoning_trace) + 1,
                    action="give_up",
                    reasoning=decision.get("reasoning", "Could not find relevant information"),
                ))
                break
        
        # Generate final answer
        answer_result = await self._generate_answer(
            question=question,
            gathered_info=gathered_info,
        )
        
        # Build citations
        citations: list[Citation] = []
        if config.generate_citations:
            for node_id, title, excerpt in gathered_info:
                node = self.doc_index.find_node(node_id)
                if node:
                    citations.append(Citation(
                        node_id=node_id,
                        node_title=title,
                        excerpt=excerpt[:500] if excerpt else "",
                        char_range=(node.text_span.start_char, node.text_span.end_char),
                    ))
        
        return QAResult(
            question=question,
            answer=answer_result.get("answer", "Could not find an answer."),
            confidence=answer_result.get("confidence", 0.0),
            citations=citations,
            reasoning_trace=[step.to_dict() for step in reasoning_trace],
            nodes_visited=visited_nodes,
        )
    
    async def answer_with_progress(
        self,
        question: str,
        config: Optional[AgenticQAConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> QAResult:
        """Answer question with progress reporting"""
        config = config or AgenticQAConfig()
        
        reporter = ProgressReporter(
            operation=OperationType.QA,
            total_steps=config.max_iterations + 2,  # +2 for plan and answer
            callback=progress_callback,
        )
        
        # Plan
        reporter.report("Planning", "Analyzing question and document structure")
        plan = await self._plan_search(question)
        
        # Search iterations
        visited_nodes: list[str] = []
        gathered_info: list[tuple[str, str, str]] = []
        reasoning_trace: list[ReasoningStep] = []
        
        for i in range(config.max_iterations):
            reporter.report(f"Searching ({i+1}/{config.max_iterations})", 
                          f"Visited {len(visited_nodes)} nodes")
            
            structure_view = self._get_structure_view(visited_nodes)
            decision = await self._decide_next_action(question, structure_view, gathered_info, plan)
            
            action = decision.get("action", "give_up")
            
            if action == "answer" or action == "give_up":
                break
            
            if action == "read_section":
                node_id = decision.get("node_id")
                if node_id and node_id not in visited_nodes:
                    visited_nodes.append(node_id)
                    node = self.doc_index.find_node(node_id)
                    if node:
                        text = self.doc_index.get_node_text(node_id)
                        if text:
                            extraction = await self._extract_relevant_info(question, text)
                            if extraction.get("found_relevant"):
                                gathered_info.append((
                                    node_id,
                                    node.title,
                                    extraction.get("excerpt", ""),
                                ))
        
        # Generate answer
        reporter.report("Generating Answer", "Synthesizing final answer")
        answer_result = await self._generate_answer(question, gathered_info)
        
        # Build citations
        citations = []
        for node_id, title, excerpt in gathered_info:
            node = self.doc_index.find_node(node_id)
            if node:
                citations.append(Citation(
                    node_id=node_id,
                    node_title=title,
                    excerpt=excerpt[:500],
                    char_range=(node.start_char, node.end_char),
                ))
        
        return QAResult(
            question=question,
            answer=answer_result.get("answer", "Could not find an answer."),
            confidence=answer_result.get("confidence", 0.0),
            citations=citations,
            reasoning_trace=[s.to_dict() for s in reasoning_trace],
            nodes_visited=visited_nodes,
        )
    
    async def answer_stream(
        self,
        question: str,
        config: Optional[AgenticQAConfig] = None,
    ) -> StreamingQAResult:
        """
        Answer question with streaming response.
        
        Returns immediately with a StreamingQAResult containing
        an async iterator for the answer.
        """
        config = config or AgenticQAConfig()
        
        # Do the search first (non-streaming)
        visited_nodes: list[str] = []
        gathered_info: list[tuple[str, str, str]] = []
        reasoning_trace: list[ReasoningStep] = []
        
        plan = await self._plan_search(question)
        
        for _ in range(config.max_iterations):
            structure_view = self._get_structure_view(visited_nodes)
            decision = await self._decide_next_action(question, structure_view, gathered_info, plan)
            
            action = decision.get("action", "give_up")
            
            if action in ("answer", "give_up"):
                break
            
            if action == "read_section":
                node_id = decision.get("node_id")
                if node_id and node_id not in visited_nodes:
                    visited_nodes.append(node_id)
                    node = self.doc_index.find_node(node_id)
                    if node:
                        text = self.doc_index.get_node_text(node_id)
                        if text:
                            extraction = await self._extract_relevant_info(question, text)
                            if extraction.get("found_relevant"):
                                gathered_info.append((
                                    node_id,
                                    node.title,
                                    extraction.get("excerpt", ""),
                                ))
        
        # Create streaming answer generator
        async def generate_answer_stream() -> AsyncIterator[StreamChunk]:
            context = self._build_context(gathered_info)
            
            prompt = f"""Answer this question based on the document excerpts.

Question: {question}

{context}

Provide a clear, accurate answer based only on the information above.
If the information is incomplete, acknowledge what's missing."""

            async for chunk in self.llm.complete_stream(prompt):
                yield chunk
        
        # Build citations
        citations = []
        for node_id, title, excerpt in gathered_info:
            node = self.doc_index.find_node(node_id)
            if node:
                citations.append(Citation(
                    node_id=node_id,
                    node_title=title,
                    excerpt=excerpt[:500],
                    char_range=(node.start_char, node.end_char),
                ))
        
        return StreamingQAResult(
            question=question,
            answer_stream=generate_answer_stream(),
            reasoning_trace=[s.to_dict() for s in reasoning_trace],
            nodes_visited=visited_nodes,
            citations=citations,
        )
    
    async def _plan_search(self, question: str) -> dict:
        """Plan the search strategy"""
        structure_summary = self._get_structure_summary()
        
        prompt = f"""Plan how to find the answer to this question in the document.

Document: {self.doc_index.description or self.doc_index.doc_name}
Document Type: {self.doc_index.doc_type.value}

Document Structure:
{structure_summary}

Question: {question}

Think about:
1. What type of information is needed?
2. Which sections likely contain this information?
3. Are there multiple places to check?

Return JSON:
{{
  "reasoning": "explanation of search strategy",
  "targets": ["section types or titles to look for"],
  "priority": "high/medium/low confidence this can be answered"
}}"""

        try:
            return await self.llm.complete_json(prompt)
        except Exception as e:
            logger.warning(f"Plan generation failed: {e}")
            return {"reasoning": "Search all sections", "targets": [], "priority": "medium"}
    
    async def _decide_next_action(
        self,
        question: str,
        structure: str,
        gathered_info: list,
        plan: dict,
    ) -> dict:
        """Decide the next action to take"""
        info_summary = ""
        if gathered_info:
            info_summary = "Information gathered so far:\n"
            for node_id, title, excerpt in gathered_info:
                excerpt_preview = excerpt[:200] if excerpt else "No specific excerpt"
                info_summary += f"- [{node_id}] {title}: {excerpt_preview}...\n"
        else:
            info_summary = "No information gathered yet.\n"
        
        prompt = f"""Decide the next action to answer this question.

Question: {question}

Document Structure (node_id in brackets, [visited] marks already read sections):
{structure}

{info_summary}

What should we do next?

Options:
1. "read_section" - Read a specific section (provide node_id)
2. "answer" - We have enough information to answer
3. "give_up" - Cannot find the answer in this document

Return JSON:
{{
  "action": "read_section",
  "node_id": "0001",
  "reasoning": "why this action"
}}

Choose "answer" if you have sufficient information. Choose a section that hasn't been visited yet."""

        try:
            return await self.llm.complete_json(prompt)
        except Exception as e:
            logger.warning(f"Decision failed: {e}")
            return {"action": "give_up", "reasoning": "Decision error"}
    
    async def _extract_relevant_info(self, question: str, text: str) -> dict:
        """Extract relevant information from section text"""
        # Truncate if too long
        if len(text) > 10000:
            text = text[:5000] + "\n...[truncated]...\n" + text[-5000:]
        
        prompt = f"""Extract information relevant to answering this question.

Question: {question}

Section Text:
{text}

Return JSON:
{{
  "found_relevant": true/false,
  "excerpt": "the specific relevant text (quote key parts)",
  "summary": "what this tells us about the answer",
  "follow_refs": true/false
}}

Set found_relevant to false if this section doesn't help answer the question."""

        try:
            return await self.llm.complete_json(prompt)
        except Exception as e:
            logger.warning(f"Extraction failed: {e}")
            return {"found_relevant": False, "summary": "Extraction error"}
    
    async def _generate_answer(
        self,
        question: str,
        gathered_info: list,
    ) -> dict:
        """Generate final answer from gathered information"""
        if not gathered_info:
            return {
                "answer": "I could not find information to answer this question in the document.",
                "confidence": 0.0,
            }
        
        context = self._build_context(gathered_info)
        
        prompt = f"""Answer this question based on the document excerpts.

Question: {question}

{context}

Provide a clear, accurate answer based only on the information above.
If the information is incomplete, acknowledge what's missing.

Return JSON:
{{
  "answer": "your complete answer",
  "confidence": 0.0-1.0
}}

Confidence guidelines:
- 0.9-1.0: Complete, well-supported answer
- 0.7-0.8: Good answer with minor gaps
- 0.5-0.6: Partial answer
- Below 0.5: Very incomplete"""

        try:
            return await self.llm.complete_json(prompt)
        except Exception as e:
            logger.warning(f"Answer generation failed: {e}")
            return {"answer": "Error generating answer.", "confidence": 0.0}
    
    def _build_context(self, gathered_info: list) -> str:
        """Build context string from gathered info"""
        if not gathered_info:
            return "No relevant information found."
        
        context = "Relevant information from the document:\n\n"
        for node_id, title, excerpt in gathered_info:
            context += f"From [{node_id}] {title}:\n{excerpt}\n\n"
        return context
    
    def _get_structure_view(self, visited: list[str]) -> str:
        """Get structure view with visited markers"""
        def node_to_text(node: TreeNode, depth: int = 0) -> str:
            indent = "  " * depth
            visited_mark = " [visited]" if node.node_id in visited else ""
            summary = f" - {node.summary[:60]}..." if node.summary else ""
            
            lines = [f"{indent}[{node.node_id}] {node.title}{summary}{visited_mark}"]
            for child in node.children:
                lines.append(node_to_text(child, depth + 1))
            
            return "\n".join(lines)
        
        return "\n".join(node_to_text(n) for n in self.doc_index.structure)
    
    def _get_structure_summary(self) -> str:
        """Get concise structure summary"""
        def summarize_node(node: TreeNode, depth: int = 0) -> str:
            if depth > 2:  # Limit depth
                return ""
            indent = "  " * depth
            lines = [f"{indent}- {node.title}"]
            for child in node.children[:5]:  # Limit children shown
                child_text = summarize_node(child, depth + 1)
                if child_text:
                    lines.append(child_text)
            if len(node.children) > 5:
                lines.append(f"{indent}  ... and {len(node.children) - 5} more")
            return "\n".join(lines)
        
        return "\n".join(summarize_node(n) for n in self.doc_index.structure)


# ============================================================================
# Convenience function
# ============================================================================

async def answer_question(
    doc_index: DocumentIndex,
    question: str,
    model: str = "gpt-4o",
    max_iterations: int = 5,
) -> QAResult:
    """
    Convenience function to answer a question.
    
    Args:
        doc_index: Document index to search
        question: Question to answer
        model: LLM model to use
        max_iterations: Max search iterations
    
    Returns:
        QAResult
    """
    qa = AgenticQA(doc_index, llm_config=LLMConfig(model=model))
    config = AgenticQAConfig(max_iterations=max_iterations)
    return await qa.answer(question, config)
