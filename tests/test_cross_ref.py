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
        """Create a test document index with known structure"""
        text = "x" * 1000
        
        nodes = [
            TreeNode(
                node_id="0001",
                title="PART I",
                level=0,
                text_span=TextSpan(0, 300, 0, 3),
            ),
            TreeNode(
                node_id="0002",
                title="Item 1. Business",
                level=1,
                text_span=TextSpan(0, 100, 0, 1),
            ),
            TreeNode(
                node_id="0003",
                title="Item 1A. Risk Factors",
                level=1,
                text_span=TextSpan(100, 200, 1, 2),
            ),
            TreeNode(
                node_id="0004",
                title="Note 15 - Segment Information",
                level=2,
                text_span=TextSpan(300, 400, 3, 4),
            ),
            TreeNode(
                node_id="0005",
                title="Appendix G - Product List",
                level=2,
                text_span=TextSpan(400, 500, 4, 5),
            ),
        ]
        nodes[0].children = [nodes[1], nodes[2]]
        
        return DocumentIndex(
            doc_id="test",
            doc_name="test",
            doc_type=DocumentType.SEC_10K,
            original_text=text,
            chunks=["x" * 100] * 10,
            chunk_char_offsets=[(i*100, (i+1)*100) for i in range(10)],
            structure=nodes,
        )
    
    def test_resolve_item_reference(self):
        doc_index = self._create_test_index()
        
        # Add a reference to resolve
        ref = CrossReference(
            source_node_id="0002",
            target_description="Item 1A",
            reference_text="See Item 1A for risks.",
        )
        doc_index.structure[0].children[0].cross_references = [ref]
        
        resolver = CrossReferenceResolver(CrossRefConfig(use_llm_resolution=False))
        resolver.resolve_references_sync(doc_index)
        
        # Check resolution
        resolved_ref = doc_index.structure[0].children[0].cross_references[0]
        assert resolved_ref.resolved is True
        assert resolved_ref.target_node_id == "0003"
    
    def test_resolve_note_reference(self):
        doc_index = self._create_test_index()
        
        ref = CrossReference(
            source_node_id="0002",
            target_description="Note 15",
            reference_text="See Note 15.",
        )
        doc_index.structure[0].children[0].cross_references = [ref]
        
        resolver = CrossReferenceResolver(CrossRefConfig(use_llm_resolution=False))
        resolver.resolve_references_sync(doc_index)
        
        resolved_ref = doc_index.structure[0].children[0].cross_references[0]
        assert resolved_ref.resolved is True
        assert resolved_ref.target_node_id == "0004"
    
    def test_resolve_appendix_reference(self):
        doc_index = self._create_test_index()
        
        ref = CrossReference(
            source_node_id="0002",
            target_description="Appendix G",
            reference_text="See Appendix G.",
        )
        doc_index.structure[0].children[0].cross_references = [ref]
        
        resolver = CrossReferenceResolver(CrossRefConfig(use_llm_resolution=False))
        resolver.resolve_references_sync(doc_index)
        
        resolved_ref = doc_index.structure[0].children[0].cross_references[0]
        assert resolved_ref.resolved is True
        assert resolved_ref.target_node_id == "0005"


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
