import pytest
from nodes.includes.mapping_utils import get_choices_for

def test_get_choices_for_empty():
    assert get_choices_for([]) == ["none", "random", "random2"]
    assert get_choices_for(None) == ["none", "random", "random2"]

def test_get_choices_for_dict():
    test_dict = {"rock": 1, "pop": 2}
    result = get_choices_for(test_dict)
    assert result == ["none", "random", "random2", "pop", "rock"]

def test_get_choices_for_wildcards():
    items = ["__ROCK__", "__POP__", "__JAZZ__"]
    result = get_choices_for(items)
    assert result == ["none", "random", "random2", "(jazz)", "(pop)", "(rock)"]

def test_get_choices_for_sorting():
    items = ["zebra", "_apple", "CHERRY"]
    result = get_choices_for(items)
    # Order: _apple, CHERRY, zebra
    assert result == ["none", "random", "random2", "_apple", "CHERRY", "zebra"]
