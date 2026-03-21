import pytest
from nodes.lyrics_formatter_node import AceStepLyricsFormatter

@pytest.fixture
def formatter():
    return AceStepLyricsFormatter()

def test_format_adds_intro_outro(formatter):
    lyrics = "Line 1\nLine 2"
    result = formatter.format(lyrics)[0]
    
    lines = result.split("\n")
    assert lines[0] == "[Intro]"
    assert lines[1] == "Line 1"
    assert lines[-1] == "[Outro]"

def test_format_preserves_existing_tags(formatter):
    lyrics = "[Intro]\nLine 1\n[Outro]"
    result = formatter.format(lyrics)[0]
    
    assert result.count("[Intro]") == 1
    assert result.count("[Outro]") == 1

def test_format_line_wrap_80_chars(formatter):
    # 85 chars, should wrap
    long_line = "A" * 40 + " " + "B" * 44
    result = formatter.format(long_line)[0]
    
    lines = result.split("\n")
    # Intro, A*40, B*44, empty, Outro
    assert lines[1] == "A" * 40
    assert lines[2] == "B" * 44
