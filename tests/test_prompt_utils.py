import pytest
import random
from nodes.includes.prompt_utils import get_keyscales, sort_weighted, expand_wildcards

def test_get_keyscales():
    scales = get_keyscales()
    assert "Auto-decide" in scales
    assert "C major" in scales
    # Overridden: A minor with accidental is skipped
    assert "Ab minor" not in scales
    assert "A minor" in scales
    assert "Bb minor" in scales

def test_sort_weighted(monkeypatch):
    # Mocking _COMPONENT_WEIGHTS by monkeypatching the module's dictionary
    import nodes.includes.prompt_utils as pu
    monkeypatch.setitem(pu._COMPONENT_WEIGHTS, "GENRES", 10.0)
    monkeypatch.setitem(pu._COMPONENT_WEIGHTS, "MOODS", 5.0)
    monkeypatch.setitem(pu._COMPONENT_WEIGHTS, "INSTRUMENTS", 0.0)
    
    names = ["MOODS", "INSTRUMENTS", "GENRES"]
    sorted_names = sort_weighted(names)
    
    # GENRES (10.0) > MOODS (5.0) > INSTRUMENTS (0.0)
    assert sorted_names == ["GENRES", "MOODS", "INSTRUMENTS"]

def test_expand_wildcards(monkeypatch):
    import nodes.includes.prompt_utils as pu
    
    # Mock get_component to return fixed values for specific wildcards
    def mock_get_component(name, default=None):
        mapping = {
            "COLOR": ["red", "blue"],
            "ANIMAL": ["cat", "dog"]
        }
        return mapping.get(name.upper())

    monkeypatch.setattr(pu, "get_component", mock_get_component)
    
    rng = random.Random(42) # Deterministic
    
    # Case 1: Simple expansion
    text = "A __COLOR__ __ANIMAL__"
    # Random(42).choice(["red", "blue"]) -> "blue"
    # Random(42).choice(["cat", "dog"]) -> "dog"
    expanded = expand_wildcards(text, rng)
    assert "blue" in expanded or "red" in expanded
    assert "cat" in expanded or "dog" in expanded
    
    # Case 2: No wildcards
    assert expand_wildcards("No wildcards here", rng) == "No wildcards here"
    
    # Case 3: Missing component returns original
    assert "__MISSING__" in expand_wildcards("__MISSING__", rng)
