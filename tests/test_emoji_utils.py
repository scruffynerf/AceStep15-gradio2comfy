import pytest
from unittest.mock import MagicMock
from nodes.includes.emoji_utils import _make_drawing_bw

def test_make_drawing_bw():
    # Mock a ReportLab drawing object
    mock_obj = MagicMock()
    mock_obj.fillColor = "some_color"
    mock_obj.strokeColor = "some_color"
    mock_obj.contents = [] # No recursion for now
    
    # Test "white_solid"
    from reportlab.graphics import shapes
    _make_drawing_bw(mock_obj, mode="white_solid")
    assert mock_obj.fillColor == shapes.colors.white
    assert mock_obj.strokeColor == shapes.colors.white

    # Test "white_outline"
    mock_obj.fillColor = "some_color"
    mock_obj.strokeColor = "some_color"
    _make_drawing_bw(mock_obj, mode="white_outline")
    assert mock_obj.fillColor == shapes.colors.black
    assert mock_obj.strokeColor == shapes.colors.white

def test_make_drawing_bw_recursion():
    from reportlab.graphics import shapes
    
    child = MagicMock()
    child.fillColor = "blue"
    
    parent = MagicMock()
    parent.contents = [child]
    # Delete fillColor from parent to test recursion only
    del parent.fillColor
    
    _make_drawing_bw(parent, mode="white_solid")
    assert child.fillColor == shapes.colors.white
