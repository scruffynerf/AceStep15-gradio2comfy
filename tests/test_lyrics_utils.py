import pytest
from nodes.includes.lyrics_utils import clean_markdown_formatting, safe_filename

def test_clean_markdown_formatting():
    # Test <think> block removal
    text_with_think = "<think>I should write a poem about cats.</think>The cat sat on the mat."
    assert clean_markdown_formatting(text_with_think) == "The cat sat on the mat."
    
    # Test case-insensitive <think>
    text_with_think_caps = "<THINK>Deep thoughts.</THINK>Normalized text."
    assert clean_markdown_formatting(text_with_think_caps) == "Normalized text."
    
    # Test code fence removal
    text_with_fences = "```\n[Verse]\nLyrics here\n```"
    assert clean_markdown_formatting(text_with_fences) == "[Verse]\nLyrics here"
    
    # Test tag normalization [Verse 1] -> [Verse]
    text_with_numbered_tags = "[Verse 1]\nLine 1\n[Chorus 2]\nLine 2"
    result = clean_markdown_formatting(text_with_numbered_tags)
    assert "[Verse]" in result
    assert "[Verse 1]" not in result
    assert "[Chorus]" in result
    assert "[Chorus 2]" not in result

def test_safe_filename():
    assert safe_filename("Artist / Title?") == "Artist_Title"
    assert safe_filename("My Song (Remix)") == "My_Song_Remix"
    assert safe_filename("Space   and---Dashes") == "Space_and_Dashes"
    assert safe_filename("...Leading Dots") == "...Leading_Dots"
