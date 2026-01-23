"""
Tests for cross-reference detection and resolution.
"""

import pytest

from documentindex.cross_ref import (
    CrossReferenceDetector,
    CrossReferenceResolver,
    CrossReferenceFollower,
    CrossRefConfig,
)
from documentindex.models import (
    DocumentIndex, TreeNode, TextSpan, DocumentType, CrossReference
)


class TestCrossReferenceDetector:
    """Tests for cross-reference detection"""
    
    def test_detect_appendix_reference(self):
        detector = CrossReferenceDetector()
        text = "For a complete list, see Appendix G."
        
        refs = detector.detect_references(text, "0001")
        
        assert len(refs) >= 1
        assert any("Appendix G" in r.target_description for r in refs)
    
    def test_detect_note_reference(self):
        detector = CrossReferenceDetector()
        text = "Refer to Note 15 for segment information. (Note 12)"
        
        refs = detector.detect_references(text, "0001")
        
        assert len(refs) >= 1
        targets = [r.target_description for r in refs]
        assert any("15" in t for t in targets)
    
    def test_detect_item_reference(self):
        detector = CrossReferenceDetector()
        text = "As described in Item 1A, risk factors include..."
        
        refs = detector.detect_references(text, "0001")
        
        assert len(refs) >= 1
        assert any("Item 1A" in r.target_description for r in refs)
    
    def test_detect_table_reference(self):
        detector = CrossReferenceDetector()
        text = "Revenue breakdown is shown in Table 5.3."
        
        refs = detector.detect_references(text, "0001")
        
        assert len(refs) >= 1
        assert any("Table" in r.target_description for r in refs)
    
    def test_detect_section_reference(self):
        detector = CrossReferenceDetector()
        text = "See Section 3.1 for more details."
        
        refs = detector.detect_references(text, "0001")
        
        assert len(refs) >= 1
        assert any("Section" in r.target_description for r in refs)
    
    def test_detect_exhibit_reference(self):
        detector = CrossReferenceDetector()
        text = "Refer to Exhibit 10.1 for the agreement."
        
        refs = detector.detect_references(text, "0001")
        
        assert len(refs) >= 1
        assert any("Exhibit" in r.target_description for r in refs)
    
    def test_detect_part_reference(self):
        detector = CrossReferenceDetector()
        text = "See Part II for financial statements."
        
        refs = detector.detect_references(text, "0001")
        
        assert len(refs) >= 1
        assert any("Part" in r.target_description for r in refs)
    
    def test_detect_multiple_references(self):
        detector = CrossReferenceDetector()
        text = """
        As shown in Table 1, revenue increased. 
        See Note 5 for accounting policies.
        Refer to Appendix A for definitions.
        """
        
        refs = detector.detect_references(text, "0001")
        
        # Should find multiple references
        assert len(refs) >= 2
    
    def test_no_duplicate_references(self):
        detector = CrossReferenceDetector()
        text = "See Note 15. Also refer to Note 15 again."
        
        refs = detector.detect_references(text, "0001")
        
        # Should deduplicate
        note_15_refs = [r for r in refs if "15" in r.target_description]
        assert len(note_15_refs) == 1
    
    def test_empty_text(self):
        detector = CrossReferenceDetector()
        refs = detector.detect_references("", "0001")
        assert refs == []
    
    def test_no_references(self):
        detector = CrossReferenceDetector()
        text = "This is plain text with no cross-references."
        refs = detector.detect_references(text, "0001")
        assert refs == []


class TestCrossReferenceResolver:
    """Tests for cross-reference resolution"""
    
    def _create_test_index(self) -> DocumentIndex:
        """Create a test document index with known structure and references in text"""
        # Create text that contains cross-references
        text_part1 = "PART I\n\nThis section discusses business. See Item 1A for risk factors.\n"
        text_item1 = "Item 1. Business\n\nBusiness description here. Refer to Note 15 for details.\n"
        text_item1a = "Item 1A. Risk Factors\n\nRisk factors described here.\n"
        text_note15 = "Note 15 - Segment Information\n\nSegment details here. See Appendix G.\n"
        text_appendix = "Appendix G - Product List\n\nProduct list here.\n"
        
        full_text = text_part1 + text_item1 + text_item1a + text_note15 + text_appendix
        
        # Calculate offsets
        offset1 = 0
        offset2 = len(text_part1)
        offset3 = offset2 + len(text_item1)
        offset4 = offset3 + len(text_item1a)
        offset5 = offset4 + len(text_note15)
        
        # Create nodes with proper parent-child relationships
        part1 = TreeNode(
            node_id="0001",
            title="PART I",
            level=0,
            text_span=TextSpan(offset1, offset3, 0, 2),
        )
        item1 = TreeNode(
            node_id="0002",
            title="Item 1. Business",
            level=1,
            text_span=TextSpan(offset2, offset3, 1, 2),
            parent_id="0001",
        )
        item1a = TreeNode(
            node_id="0003",
            title="Item 1A. Risk Factors",
            level=1,
            text_span=TextSpan(offset3, offset4, 2, 3),
            parent_id="0001",
        )
        note15 = TreeNode(
            node_id="0004",
            title="Note 15 - Segment Information",
            level=0,
            text_span=TextSpan(offset4, offset5, 3, 4),
        )
        appendix_g = TreeNode(
            node_id="0005",
            title="Appendix G - Product List",
            level=0,
            text_span=TextSpan(offset5, len(full_text), 4, 5),
        )
        
        # Set up children
        part1.children = [item1, item1a]
        
        # Create chunks matching the text
        chunks = [text_part1, text_item1, text_item1a, text_note15, text_appendix]
        chunk_offsets = [
            (offset1, offset2),
            (offset2, offset3),
            (offset3, offset4),
            (offset4, offset5),
            (offset5, len(full_text)),
        ]
        
        return DocumentIndex(
            doc_id="test",
            doc_name="test",
            doc_type=DocumentType.SEC_10K,
            original_text=full_text,
            chunks=chunks,
            chunk_char_offsets=chunk_offsets,
            structure=[part1, note15, appendix_g],
        )
    
    def test_resolve_item_reference(self):
        """Test that Item 1A reference in PART I text is resolved"""
        doc_index = self._create_test_index()
        
        resolver = CrossReferenceResolver(CrossRefConfig(use_llm_resolution=False))
        all_refs = resolver.resolve_references_sync(doc_index)
        
        # Find the reference to Item 1A
        item_refs = [r for r in all_refs if "Item 1A" in r.target_description or "item 1a" in r.target_description.lower()]
        
        assert len(item_refs) >= 1, f"Expected Item 1A reference, found: {[r.target_description for r in all_refs]}"
        ref = item_refs[0]
        assert ref.resolved is True
        assert ref.target_node_id == "0003"
    
    def test_resolve_note_reference(self):
        """Test that Note 15 reference in Item 1 text is resolved"""
        doc_index = self._create_test_index()
        
        resolver = CrossReferenceResolver(CrossRefConfig(use_llm_resolution=False))
        all_refs = resolver.resolve_references_sync(doc_index)
        
        # Find the reference to Note 15
        note_refs = [r for r in all_refs if "Note 15" in r.target_description or "note 15" in r.target_description.lower()]
        
        assert len(note_refs) >= 1, f"Expected Note 15 reference, found: {[r.target_description for r in all_refs]}"
        ref = note_refs[0]
        assert ref.resolved is True
        assert ref.target_node_id == "0004"
    
    def test_resolve_appendix_reference(self):
        """Test that Appendix G reference in Note 15 text is resolved"""
        doc_index = self._create_test_index()
        
        resolver = CrossReferenceResolver(CrossRefConfig(use_llm_resolution=False))
        all_refs = resolver.resolve_references_sync(doc_index)
        
        # Find the reference to Appendix G
        appendix_refs = [r for r in all_refs if "Appendix G" in r.target_description or "appendix g" in r.target_description.lower()]
        
        assert len(appendix_refs) >= 1, f"Expected Appendix G reference, found: {[r.target_description for r in all_refs]}"
        ref = appendix_refs[0]
        assert ref.resolved is True
        assert ref.target_node_id == "0005"


class TestCrossReferenceFollower:
    """Tests for cross-reference following"""
    
    def test_get_referenced_nodes(self):
        # Create simple structure
        text = "x" * 200
        node1 = TreeNode(
            node_id="0001",
            title="Section 1",
            level=0,
            text_span=TextSpan(0, 100, 0, 1),
        )
        node2 = TreeNode(
            node_id="0002", 
            title="Section 2",
            level=0,
            text_span=TextSpan(100, 200, 1, 2),
        )
        
        # Add resolved cross-reference
        node1.cross_references = [
            CrossReference(
                source_node_id="0001",
                target_description="Section 2",
                target_node_id="0002",
                resolved=True,
            )
        ]
        
        doc_index = DocumentIndex(
            doc_id="test",
            doc_name="test",
            doc_type=DocumentType.GENERIC,
            original_text=text,
            chunks=["x" * 100, "x" * 100],
            chunk_char_offsets=[(0, 100), (100, 200)],
            structure=[node1, node2],
        )
        
        follower = CrossReferenceFollower(doc_index)
        referenced = follower.get_referenced_nodes("0001")
        
        assert len(referenced) == 1
        assert referenced[0].node_id == "0002"
    
    def test_get_referencing_nodes(self):
        text = "x" * 200
        node1 = TreeNode(
            node_id="0001",
            title="Section 1",
            level=0,
            text_span=TextSpan(0, 100, 0, 1),
        )
        node2 = TreeNode(
            node_id="0002",
            title="Section 2",
            level=0,
            text_span=TextSpan(100, 200, 1, 2),
        )
        
        node1.cross_references = [
            CrossReference(
                source_node_id="0001",
                target_description="Section 2",
                target_node_id="0002",
                resolved=True,
            )
        ]
        
        doc_index = DocumentIndex(
            doc_id="test",
            doc_name="test",
            doc_type=DocumentType.GENERIC,
            original_text=text,
            chunks=["x" * 100, "x" * 100],
            chunk_char_offsets=[(0, 100), (100, 200)],
            structure=[node1, node2],
        )
        
        follower = CrossReferenceFollower(doc_index)
        referencing = follower.get_referencing_nodes("0002")
        
        assert len(referencing) == 1
        assert referencing[0].node_id == "0001"
    
    def test_get_reference_graph(self):
        text = "x" * 300
        node1 = TreeNode(node_id="0001", title="S1", level=0, text_span=TextSpan(0, 100, 0, 1))
        node2 = TreeNode(node_id="0002", title="S2", level=0, text_span=TextSpan(100, 200, 1, 2))
        node3 = TreeNode(node_id="0003", title="S3", level=0, text_span=TextSpan(200, 300, 2, 3))
        
        node1.cross_references = [
            CrossReference("0001", "S2", "0002", "", True),
            CrossReference("0001", "S3", "0003", "", True),
        ]
        node2.cross_references = [
            CrossReference("0002", "S3", "0003", "", True),
        ]
        
        doc_index = DocumentIndex(
            doc_id="test",
            doc_name="test",
            doc_type=DocumentType.GENERIC,
            original_text=text,
            chunks=["x" * 100] * 3,
            chunk_char_offsets=[(0, 100), (100, 200), (200, 300)],
            structure=[node1, node2, node3],
        )
        
        follower = CrossReferenceFollower(doc_index)
        graph = follower.get_reference_graph()
        
        assert "0001" in graph
        assert set(graph["0001"]) == {"0002", "0003"}
        assert "0002" in graph
        assert graph["0002"] == ["0003"]
