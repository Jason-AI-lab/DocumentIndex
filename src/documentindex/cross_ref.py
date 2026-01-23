"""
Cross-reference detection and resolution.

Detects references like:
- "See Appendix G"
- "Refer to Note 15"
- "As shown in Table 5.3"
- "Described in Item 1A"

And resolves them to actual node IDs in the document tree.
"""

from dataclasses import dataclass
from typing import Optional
import re
import logging

from .models import DocumentIndex, TreeNode, CrossReference
from .llm_client import LLMClient, LLMConfig

logger = logging.getLogger(__name__)


@dataclass
class CrossRefConfig:
    """Configuration for cross-reference resolution"""
    use_llm_resolution: bool = False  # Use LLM for ambiguous references
    llm_config: Optional[LLMConfig] = None
    max_llm_refs: int = 20  # Max refs to resolve with LLM in one batch


class CrossReferenceDetector:
    """Detects cross-references in text"""
    
    # Patterns for detecting cross-references
    REFERENCE_PATTERNS = [
        # Appendix references
        (r"(?:see|refer(?:red)?\s+to|in|described\s+in|set\s+forth\s+in)\s+Appendix\s+([A-Z](?:\d+)?)", "appendix"),
        
        # Note references (financial statements)
        (r"(?:see|refer(?:red)?\s+to)\s+Note\s+(\d+(?:\.\d+)?)", "note"),
        (r"\(Note\s+(\d+(?:\.\d+)?)\)", "note"),
        
        # Item references (SEC filings)
        (r"(?:see|refer(?:red)?\s+to|in|described\s+in)\s+Item\s+(\d+[A-Z]?)", "item"),
        
        # Table references
        (r"(?:see|in|shown\s+in|refer(?:red)?\s+to)\s+Table\s+(\d+(?:\.\d+)?)", "table"),
        
        # Figure references
        (r"(?:see|in|shown\s+in)\s+Figure\s+(\d+(?:\.\d+)?)", "figure"),
        
        # Section references
        (r"(?:see|refer(?:red)?\s+to|in)\s+Section\s+(\d+(?:\.\d+)*)", "section"),
        
        # Exhibit references
        (r"(?:see|refer(?:red)?\s+to)\s+Exhibit\s+(\d+(?:\.\d+)?)", "exhibit"),
        
        # Part references
        (r"(?:see|in)\s+Part\s+([IVX]+|\d+)", "part"),
        
        # Page references (less common but useful)
        (r"(?:see\s+)?page\s+(\d+)", "page"),
    ]
    
    def __init__(self):
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), ref_type)
            for pattern, ref_type in self.REFERENCE_PATTERNS
        ]
    
    def detect_references(self, text: str, source_node_id: str = "") -> list[CrossReference]:
        """
        Detect all cross-references in text.
        
        Args:
            text: Text to scan for references
            source_node_id: Node ID where references are found
        
        Returns:
            List of CrossReference objects
        """
        references = []
        seen = set()  # Avoid duplicates
        
        for pattern, ref_type in self._compiled_patterns:
            for match in re.finditer(pattern, text):
                full_match = match.group(0)
                ref_target = match.group(1)
                
                # Create unique key for deduplication
                key = f"{ref_type}:{ref_target}"
                if key in seen:
                    continue
                seen.add(key)
                
                # Get surrounding context
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                # Format target description
                target_desc = f"{ref_type.title()} {ref_target}"
                
                references.append(CrossReference(
                    source_node_id=source_node_id,
                    target_description=target_desc,
                    reference_text=context.strip(),
                    resolved=False,
                ))
        
        return references


class CrossReferenceResolver:
    """Resolves cross-references to their target nodes"""
    
    def __init__(self, config: Optional[CrossRefConfig] = None):
        self.config = config or CrossRefConfig()
        self.detector = CrossReferenceDetector()
        self.llm: Optional[LLMClient] = None
        if self.config.use_llm_resolution:
            self.llm = LLMClient(self.config.llm_config or LLMConfig())
    
    async def resolve_references(
        self,
        doc_index: DocumentIndex,
    ) -> list[CrossReference]:
        """
        Resolve all cross-references in a document index.
        
        Args:
            doc_index: Document index with structure
        
        Returns:
            List of all cross-references (resolved and unresolved)
        """
        all_references: list[CrossReference] = []
        
        # Collect all references from all nodes
        for node in doc_index.get_all_nodes():
            node_text = doc_index.get_node_text(node.node_id)
            if not node_text:
                continue
            
            refs = self.detector.detect_references(node_text, node.node_id)
            all_references.extend(refs)
            node.cross_references = refs
        
        # Build lookup structures for efficient resolution
        node_lookup = self._build_node_lookup(doc_index)
        
        # Resolve each reference
        for ref in all_references:
            target_node = self._find_target_node(ref.target_description, node_lookup)
            if target_node:
                ref.target_node_id = target_node.node_id
                ref.resolved = True
        
        # Use LLM for unresolved references if enabled
        if self.config.use_llm_resolution and self.llm:
            unresolved = [r for r in all_references if not r.resolved]
            if unresolved:
                await self._resolve_with_llm(unresolved[:self.config.max_llm_refs], doc_index)
        
        doc_index.cross_references = all_references
        return all_references
    
    def resolve_references_sync(
        self,
        doc_index: DocumentIndex,
    ) -> list[CrossReference]:
        """
        Resolve cross-references synchronously (no LLM).
        
        Args:
            doc_index: Document index with structure
        
        Returns:
            List of all cross-references
        """
        all_references: list[CrossReference] = []
        
        for node in doc_index.get_all_nodes():
            node_text = doc_index.get_node_text(node.node_id)
            if not node_text:
                continue
            
            refs = self.detector.detect_references(node_text, node.node_id)
            all_references.extend(refs)
            node.cross_references = refs
        
        node_lookup = self._build_node_lookup(doc_index)
        
        for ref in all_references:
            target_node = self._find_target_node(ref.target_description, node_lookup)
            if target_node:
                ref.target_node_id = target_node.node_id
                ref.resolved = True
        
        doc_index.cross_references = all_references
        return all_references
    
    def _build_node_lookup(self, doc_index: DocumentIndex) -> dict[str, list[TreeNode]]:
        """Build lookup structures for nodes"""
        lookup: dict[str, list[TreeNode]] = {
            "by_title": [],
            "by_type": {},
        }
        
        for node in doc_index.get_all_nodes():
            lookup["by_title"].append(node)
            
            # Index by detected type
            title_lower = node.title.lower()
            for type_key in ["note", "item", "appendix", "table", "figure", "section", "exhibit", "part"]:
                if type_key in title_lower:
                    if type_key not in lookup["by_type"]:
                        lookup["by_type"][type_key] = []
                    lookup["by_type"][type_key].append(node)
        
        return lookup
    
    def _find_target_node(
        self,
        target_description: str,
        node_lookup: dict[str, list[TreeNode]],
    ) -> Optional[TreeNode]:
        """Find node matching target description"""
        # Parse target
        match = re.match(r"(\w+)\s+(.+)", target_description, re.IGNORECASE)
        if not match:
            return None
        
        ref_type = match.group(1).lower()
        ref_id = match.group(2).strip()
        
        # Look in type-specific index first
        type_nodes = node_lookup.get("by_type", {}).get(ref_type, [])
        
        for node in type_nodes:
            if self._titles_match(ref_type, ref_id, node.title):
                return node
        
        # Fall back to searching all nodes
        for node in node_lookup.get("by_title", []):
            if self._titles_match(ref_type, ref_id, node.title):
                return node
        
        return None
    
    def _titles_match(self, ref_type: str, ref_id: str, title: str) -> bool:
        """Check if target reference matches node title"""
        title_lower = title.lower()
        ref_type_lower = ref_type.lower()
        ref_id_lower = ref_id.lower()
        
        # Check if title contains both type and ID
        if ref_type_lower in title_lower:
            # Look for the ID in the title
            if ref_id_lower in title_lower:
                return True
            
            # Special handling for notes: "Note 15" might match "15. Revenue Recognition"
            if ref_type_lower == "note":
                if re.match(rf"^{re.escape(ref_id_lower)}[\.\s:]", title_lower):
                    return True
            
            # Handle items: "Item 1A" should match "Item 1A. Risk Factors"
            if ref_type_lower == "item":
                if re.search(rf"\bitem\s*{re.escape(ref_id_lower)}\b", title_lower):
                    return True
            
            # Handle appendices: "Appendix G" should match "Appendix G - Financial Tables"
            if ref_type_lower == "appendix":
                if re.search(rf"\bappendix\s*{re.escape(ref_id_lower)}\b", title_lower):
                    return True
            
            # Handle parts: "Part I" should match "PART I - FINANCIAL INFORMATION"
            if ref_type_lower == "part":
                if re.search(rf"\bpart\s*{re.escape(ref_id_lower)}\b", title_lower):
                    return True
        
        return False
    
    async def _resolve_with_llm(
        self,
        references: list[CrossReference],
        doc_index: DocumentIndex,
    ) -> None:
        """Use LLM to resolve ambiguous references"""
        if not self.llm or not references:
            return
        
        # Build node list for LLM
        node_list = []
        for node in doc_index.get_all_nodes():
            node_list.append(f"[{node.node_id}] {node.title}")
        
        nodes_text = "\n".join(node_list[:100])  # Limit to prevent context overflow
        
        # Build references list
        refs_text = "\n".join([
            f"- \"{ref.target_description}\" (context: \"{ref.reference_text[:100]}\")"
            for ref in references[:10]  # Limit batch size
        ])
        
        prompt = f"""Match these cross-references to the correct document sections.

References to match:
{refs_text}

Available sections:
{nodes_text}

Return a JSON array with the matching node_id for each reference, or null if no match:
[
  {{"target": "Note 15", "node_id": "0023"}},
  {{"target": "Appendix G", "node_id": null}}
]"""

        try:
            result = await self.llm.complete_json(prompt)
            
            if isinstance(result, list):
                result_map = {
                    r.get("target", "").lower(): r.get("node_id")
                    for r in result if isinstance(r, dict)
                }
                
                for ref in references:
                    target_key = ref.target_description.lower()
                    if target_key in result_map and result_map[target_key]:
                        ref.target_node_id = result_map[target_key]
                        ref.resolved = True
        except Exception as e:
            logger.warning(f"LLM cross-reference resolution failed: {e}")


class CrossReferenceFollower:
    """Utility to follow cross-references during retrieval"""
    
    def __init__(self, doc_index: DocumentIndex):
        self.doc_index = doc_index
    
    def get_referenced_nodes(self, node_id: str) -> list[TreeNode]:
        """Get all nodes referenced by a given node"""
        node = self.doc_index.find_node(node_id)
        if not node:
            return []
        
        referenced = []
        for ref in node.cross_references:
            if ref.resolved and ref.target_node_id:
                target = self.doc_index.find_node(ref.target_node_id)
                if target:
                    referenced.append(target)
        
        return referenced
    
    def get_referencing_nodes(self, node_id: str) -> list[TreeNode]:
        """Get all nodes that reference a given node"""
        referencing = []
        
        for node in self.doc_index.get_all_nodes():
            for ref in node.cross_references:
                if ref.target_node_id == node_id:
                    referencing.append(node)
                    break
        
        return referencing
    
    def get_reference_graph(self) -> dict[str, list[str]]:
        """Get full reference graph as adjacency list"""
        graph: dict[str, list[str]] = {}
        
        for node in self.doc_index.get_all_nodes():
            targets = []
            for ref in node.cross_references:
                if ref.resolved and ref.target_node_id:
                    targets.append(ref.target_node_id)
            if targets:
                graph[node.node_id] = targets
        
        return graph
