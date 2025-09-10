"""Test EntityFrame functionality."""

import pytest
import starlings as sl


def test_entity_frame_creation():
    """Test that EntityFrame can be created and has expected initial state."""
    frame = sl.EntityFrame()

    # Initially empty
    assert len(frame) == 0
    assert frame.collection_names() == []
    assert not frame.has_collection("test")
    assert "test" not in frame


def test_entity_frame_basic_operations():
    """Test basic EntityFrame operations."""
    frame = sl.EntityFrame()

    # Test collection methods
    assert not frame.has_collection("test")
    assert not frame.remove_collection("test")  # Removing non-existent returns False

    # String representation
    repr_str = repr(frame)
    assert "EntityFrame" in repr_str
    assert "collections=0" in repr_str


def test_entity_frame_add_collection():
    """Test that add_collection works correctly."""
    frame = sl.EntityFrame()
    edges = [(0, 1, 0.9), (1, 2, 0.8)]
    collection = sl.Collection.from_edges(edges)

    # Add collection should work now
    frame.add_collection("test", collection)

    assert len(frame) == 1
    assert frame.has_collection("test")
    assert "test" in frame
    assert "test" in frame.collection_names()


def test_entity_frame_docstring_example():
    """Test the example from the docstring works as expected."""
    # Create an empty frame
    frame = sl.EntityFrame()

    # Check collection names
    assert frame.collection_names() == []  # []
    assert len(frame) == 0  # 0

    # Now we can add collections
    edges = [(0, 1, 0.9), (1, 2, 0.8)]
    collection = sl.Collection.from_edges(edges)
    frame.add_collection("v1", collection)

    assert len(frame) == 1
    assert "v1" in frame.collection_names()


def test_collection_view_semantics():
    """Test that collections returned from EntityFrame are views."""
    frame = sl.EntityFrame()
    edges = [(0, 1, 0.9), (1, 2, 0.8), (2, 3, 0.7)]
    original_collection = sl.Collection.from_edges(edges)

    # Original collection should not be a view
    assert not original_collection.is_view()

    # Add to frame
    frame.add_collection("test", original_collection)

    # Access via dictionary syntax
    view_collection = frame["test"]

    # View should be marked as view
    assert view_collection.is_view()

    # View should have same data as original
    original_partition = original_collection.at(0.8)
    view_partition = view_collection.at(0.8)
    assert len(original_partition.entities) == len(view_partition.entities)


def test_collection_copy_from_view():
    """Test that copying a view creates an independent collection."""
    frame = sl.EntityFrame()
    edges = [("a", "b", 0.9), ("b", "c", 0.8), ("d", "e", 0.7)]
    original_collection = sl.Collection.from_edges(edges)

    # Add to frame and get view
    frame.add_collection("test", original_collection)
    view_collection = frame["test"]

    # Create copy from view
    independent_copy = view_collection.copy()

    # View should still be a view
    assert view_collection.is_view()

    # Copy should not be a view
    assert not independent_copy.is_view()

    # Both should have same data
    view_partition = view_collection.at(0.8)
    copy_partition = independent_copy.at(0.8)
    assert len(view_partition.entities) == len(copy_partition.entities)


def test_collection_copy_from_regular():
    """Test that copying a regular collection works correctly."""
    edges = [("x", "y", 0.95), ("y", "z", 0.85)]
    original = sl.Collection.from_edges(edges)

    # Original should not be a view
    assert not original.is_view()

    # Copy should also not be a view
    copy = original.copy()
    assert not copy.is_view()

    # Both should have same data
    original_partition = original.at(0.9)
    copy_partition = copy.at(0.9)
    assert len(original_partition.entities) == len(copy_partition.entities)


def test_entity_frame_getitem_keyerror():
    """Test that accessing non-existent collection raises KeyError."""
    frame = sl.EntityFrame()

    with pytest.raises(KeyError) as excinfo:
        _ = frame["nonexistent"]

    assert "nonexistent" in str(excinfo.value)


def test_entity_frame_multiple_collections():
    """Test EntityFrame with multiple collections."""
    frame = sl.EntityFrame()

    # Add multiple collections
    edges1 = [(1, 2, 0.9), (2, 3, 0.8)]
    edges2 = [("a", "b", 0.95), ("c", "d", 0.7)]

    collection1 = sl.Collection.from_edges(edges1)
    collection2 = sl.Collection.from_edges(edges2)

    frame.add_collection("numeric", collection1)
    frame.add_collection("text", collection2)

    assert len(frame) == 2
    assert set(frame.collection_names()) == {"numeric", "text"}

    # Access both collections as views
    numeric_view = frame["numeric"]
    text_view = frame["text"]

    assert numeric_view.is_view()
    assert text_view.is_view()

    # Verify they have correct data
    numeric_partition = numeric_view.at(0.85)
    text_partition = text_view.at(0.9)

    # Should have expected number of entities
    assert len(numeric_partition.entities) > 0
    assert len(text_partition.entities) > 0
