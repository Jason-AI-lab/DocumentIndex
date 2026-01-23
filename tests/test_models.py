"""
Tests for data models.
"""

import pytest
from datetime import datetime

from documentindex.models import (
    DocumentType,
    TextSpan,
    TreeNode,
    DocumentIndex,
    DocumentMetadata,
    CrossReference,
    NodeMatch,
    Citation,
    QAResult,
    ProvenanceResult,
)


class TestTextSpan:
    """Tests for TextSpan model"""
    
    def test_create_valid_span(self):
        span = TextSpan(start_char=0, end_char=100, start_chunk=0, end_chunk=2)
        assert span.start_char == 0
        assert span.end_char == 100
        assert span.start_chunk == 0
        assert span.end_chunk == 2
    
    def test_invalid_char_range(self):
        with pytest.raises(ValueError):
            TextSpan(start_char=100, end_char=50, start_chunk=0, end_chunk=1)
    
    def test_invalid_chunk_range(self):
        with pytest.raises(ValueError):
            TextSpan(start_char=0, end_char=100, start_chunk=5, end_chunk=2)
    
    def test_equal_bounds_allowed(self):
        # Empty spans should be allowed
        span = TextSpan(start_char=50, end_char=50, start_chunk=1, end_chunk=1)
        assert span.start_char == span.end_char
    
    def test_to_dict(self):
        span = TextSpan(start_char=10, end_char=200, start_chunk=1, end_chunk=5)
        d = span.to_dict()
        assert d["start_char"] == 10
        assert d["end_char"] == 200
        assert d["start_chunk"] == 1
        assert d["end_chunk"] == 5
    
    def test_from_dict(self):
        data = {"start_char": 10, "end_char": 200, "start_chunk": 1, "end_chunk": 5}
        span = TextSpan.from_dict(data)
        assert span.start_char == 10
        assert span.end_char == 200


class TestTreeNode:
    """Tests for TreeNode model"""
    
    def test_create_node(self):
        span = TextSpan(start_char=0, end_char=100, start_chunk=0, end_chunk=1)
        node = TreeNode(
            node_id="0001",
            title="Test Section",
            level=0,
            text_span=span,
        )
        assert node.node_id == "0001"
        assert node.title == "Test Section"
        assert node.level == 0
    
    def test_node_properties(self):
        span = TextSpan(start_char=50, end_char=150, start_chunk=2, end_chunk=5)
        node = TreeNode(node_id="0002", title="Test", level=1, text_span=span)
        
        assert node.start_index == 2
        assert node.end_index == 5
        assert node.start_char == 50
        assert node.end_char == 150
    
    def test_node_with_children(self):
        parent_span = TextSpan(0, 1000, 0, 10)
        child_span = TextSpan(100, 500, 1, 5)
        
        parent = TreeNode(node_id="0001", title="Parent", level=0, text_span=parent_span)
        child = TreeNode(node_id="0002", title="Child", level=1, text_span=child_span, parent_id="0001")
        parent.children.append(child)
        
        assert len(parent.children) == 1
        assert parent.children[0].parent_id == "0001"
    
    def test_to_dict(self):
        span = TextSpan(0, 100, 0, 1)
        node = TreeNode(
            node_id="0001",
            title="Test",
            level=0,
            text_span=span,
            summary="A test section",
        )
        
        d = node.to_dict()
        assert d["node_id"] == "0001"
        assert d["title"] == "Test"
        assert d["summary"] == "A test section"
        assert d["start_char"] == 0
        assert d["end_char"] == 100
    
    def test_from_dict(self):
        data = {
            "node_id": "0001",
            "title": "Test Section",
            "level": 1,
            "start_char": 0,
            "end_char": 500,
            "start_index": 0,
            "end_index": 5,
            "summary": "Test summary",
        }
        node = TreeNode.from_dict(data)
        assert node.node_id == "0001"
        assert node.title == "Test Section"
        assert node.summary == "Test summary"
    
    def test_nested_from_dict(self):
        data = {
            "node_id": "0001",
            "title": "Parent",
            "level": 0,
            "start_char": 0,
            "end_char": 1000,
            "start_index": 0,
            "end_index": 10,
            "nodes": [
                {
                    "node_id": "0002",
                    "title": "Child",
                    "level": 1,
                    "start_char": 100,
                    "end_char": 500,
                    "start_index": 1,
                    "end_index": 5,
                }
            ]
        }
        node = TreeNode.from_dict(data)
        assert len(node.children) == 1
        assert node.children[0].node_id == "0002"


class TestDocumentMetadata:
    """Tests for DocumentMetadata model"""
    
    def test_create_empty(self):
        meta = DocumentMetadata()
        assert meta.company_name is None
        assert meta.ticker is None
        assert meta.key_numbers == {}
    
    def test_create_with_values(self):
        meta = DocumentMetadata(
            company_name="ACME Corp",
            ticker="ACME",
            cik="0001234567",
            fiscal_year=2024,
            key_numbers={"revenue": "$15B"},
        )
        assert meta.company_name == "ACME Corp"
        assert meta.ticker == "ACME"
        assert meta.fiscal_year == 2024
    
    def test_to_dict(self):
        meta = DocumentMetadata(
            company_name="Test Co",
            ticker="TEST",
            filing_date=datetime(2024, 3, 15),
        )
        d = meta.to_dict()
        assert d["company_name"] == "Test Co"
        assert d["ticker"] == "TEST"
        assert d["filing_date"] == "2024-03-15T00:00:00"


class TestDocumentIndex:
    """Tests for DocumentIndex model"""
    
    def test_create_index(self):
        span = TextSpan(0, 100, 0, 1)
        node = TreeNode(node_id="0001", title="Test", level=0, text_span=span)
        
        doc_index = DocumentIndex(
            doc_id="test123",
            doc_name="test_doc",
            doc_type=DocumentType.SEC_10K,
            original_text="This is test content." * 10,
            chunks=["This is test content."] * 10,
            chunk_char_offsets=[(i*21, (i+1)*21) for i in range(10)],
            structure=[node],
        )
        
        assert doc_index.doc_id == "test123"
        assert doc_index.doc_type == DocumentType.SEC_10K
        assert len(doc_index.chunks) == 10
    
    def test_get_node_text(self):
        text = "PART I\n\nThis is section one content.\n\nPART II\n\nThis is section two."
        span1 = TextSpan(0, 37, 0, 1)
        span2 = TextSpan(38, len(text), 1, 2)
        
        node1 = TreeNode(node_id="0001", title="Part I", level=0, text_span=span1)
        node2 = TreeNode(node_id="0002", title="Part II", level=0, text_span=span2)
        
        doc_index = DocumentIndex(
            doc_id="test",
            doc_name="test",
            doc_type=DocumentType.GENERIC,
            original_text=text,
            chunks=[text[:38], text[38:]],
            chunk_char_offsets=[(0, 38), (38, len(text))],
            structure=[node1, node2],
        )
        
        text1 = doc_index.get_node_text("0001")
        assert "PART I" in text1
        assert "section one" in text1
    
    def test_find_node(self):
        span = TextSpan(0, 100, 0, 1)
        child_span = TextSpan(50, 100, 0, 1)
        
        parent = TreeNode(node_id="0001", title="Parent", level=0, text_span=span)
        child = TreeNode(node_id="0002", title="Child", level=1, text_span=child_span)
        parent.children.append(child)
        
        doc_index = DocumentIndex(
            doc_id="test",
            doc_name="test",
            doc_type=DocumentType.GENERIC,
            original_text="x" * 100,
            chunks=["x" * 100],
            chunk_char_offsets=[(0, 100)],
            structure=[parent],
        )
        
        found = doc_index.find_node("0002")
        assert found is not None
        assert found.title == "Child"
        
        not_found = doc_index.find_node("9999")
        assert not_found is None
    
    def test_get_all_nodes(self):
        span = TextSpan(0, 100, 0, 1)
        parent = TreeNode(node_id="0001", title="Parent", level=0, text_span=span)
        child1 = TreeNode(node_id="0002", title="Child1", level=1, text_span=span)
        child2 = TreeNode(node_id="0003", title="Child2", level=1, text_span=span)
        parent.children = [child1, child2]
        
        doc_index = DocumentIndex(
            doc_id="test",
            doc_name="test",
            doc_type=DocumentType.GENERIC,
            original_text="x" * 100,
            chunks=["x" * 100],
            chunk_char_offsets=[(0, 100)],
            structure=[parent],
        )
        
        all_nodes = doc_index.get_all_nodes()
        assert len(all_nodes) == 3
    
    def test_get_leaf_nodes(self):
        span = TextSpan(0, 100, 0, 1)
        parent = TreeNode(node_id="0001", title="Parent", level=0, text_span=span)
        child1 = TreeNode(node_id="0002", title="Child1", level=1, text_span=span)
        child2 = TreeNode(node_id="0003", title="Child2", level=1, text_span=span)
        parent.children = [child1, child2]
        
        doc_index = DocumentIndex(
            doc_id="test",
            doc_name="test",
            doc_type=DocumentType.GENERIC,
            original_text="x" * 100,
            chunks=["x" * 100],
            chunk_char_offsets=[(0, 100)],
            structure=[parent],
        )
        
        leaves = doc_index.get_leaf_nodes()
        assert len(leaves) == 2
        assert all(not n.children for n in leaves)
    
    def test_serialization_roundtrip(self):
        span = TextSpan(0, 100, 0, 1)
        node = TreeNode(node_id="0001", title="Test", level=0, text_span=span, summary="Test summary")
        
        original = DocumentIndex(
            doc_id="test123",
            doc_name="test_doc",
            doc_type=DocumentType.SEC_10K,
            original_text="x" * 100,
            chunks=["x" * 100],
            chunk_char_offsets=[(0, 100)],
            structure=[node],
            metadata=DocumentMetadata(company_name="Test Co", ticker="TEST"),
        )
        
        # To dict and back
        data = original.to_dict(include_text=True)
        restored = DocumentIndex.from_dict(data)
        
        assert restored.doc_id == original.doc_id
        assert restored.doc_name == original.doc_name
        assert restored.doc_type == original.doc_type
        assert len(restored.structure) == 1
        assert restored.structure[0].title == "Test"


class TestNodeMatch:
    """Tests for NodeMatch model"""
    
    def test_create_match(self):
        span = TextSpan(0, 100, 0, 1)
        node = TreeNode(node_id="0001", title="Test", level=0, text_span=span)
        
        match = NodeMatch(
            node=node,
            relevance_score=0.85,
            match_reason="Directly discusses the topic",
            matched_excerpts=["relevant text here"],
        )
        
        assert match.relevance_score == 0.85
        assert len(match.matched_excerpts) == 1
    
    def test_to_dict(self):
        span = TextSpan(0, 100, 0, 1)
        node = TreeNode(node_id="0001", title="Test Section", level=0, text_span=span)
        match = NodeMatch(node=node, relevance_score=0.9, match_reason="Relevant")
        
        d = match.to_dict()
        assert d["node_id"] == "0001"
        assert d["relevance_score"] == 0.9


class TestCitation:
    """Tests for Citation model"""
    
    def test_create_citation(self):
        citation = Citation(
            node_id="0001",
            node_title="Risk Factors",
            excerpt="Climate change may affect...",
            char_range=(1000, 1500),
        )
        
        assert citation.node_id == "0001"
        assert citation.char_range == (1000, 1500)
    
    def test_to_dict(self):
        citation = Citation(
            node_id="0001",
            node_title="Test",
            excerpt="test excerpt",
            char_range=(0, 100),
        )
        d = citation.to_dict()
        assert d["node_id"] == "0001"
        assert d["char_range"] == [0, 100]


class TestQAResult:
    """Tests for QAResult model"""
    
    def test_create_result(self):
        result = QAResult(
            question="What was the revenue?",
            answer="Revenue was $15 billion.",
            confidence=0.9,
            citations=[],
            reasoning_trace=[{"step": 1, "action": "search"}],
            nodes_visited=["0001", "0002"],
        )
        
        assert result.confidence == 0.9
        assert len(result.nodes_visited) == 2


class TestProvenanceResult:
    """Tests for ProvenanceResult model"""
    
    def test_create_result(self):
        result = ProvenanceResult(
            topic="climate risks",
            evidence=[],
            total_nodes_scanned=50,
            scan_coverage=1.0,
            summary="Found 5 relevant sections.",
        )
        
        assert result.total_nodes_scanned == 50
        assert result.scan_coverage == 1.0
