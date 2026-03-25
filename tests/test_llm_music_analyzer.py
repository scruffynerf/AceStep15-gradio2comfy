import pytest
from nodes.llm_music_analyzer_node import ScromfyAceStepMusicAnalyzer

@pytest.fixture
def analyzer():
    return ScromfyAceStepMusicAnalyzer()

def test_build_gen_kwargs(analyzer):
    # Test with temperature > 0
    kwargs_sample = analyzer._build_gen_kwargs(temperature=0.5, top_p=0.9, top_k=50, repetition_penalty=1.2, seed=42)
    assert kwargs_sample["do_sample"] is True
    assert kwargs_sample["temperature"] == 0.5
    assert kwargs_sample["top_p"] == 0.9
    assert kwargs_sample["top_k"] == 50
    assert kwargs_sample["repetition_penalty"] == 1.2
    
    # Test with temperature = 0
    kwargs_greedy = analyzer._build_gen_kwargs(temperature=0.0, top_p=1.0, top_k=0, repetition_penalty=1.0, seed=42)
    assert kwargs_greedy["do_sample"] is False
    assert "temperature" not in kwargs_greedy
    assert "top_p" not in kwargs_greedy
    assert "top_k" not in kwargs_greedy
    assert "repetition_penalty" not in kwargs_greedy

def test_clean_tags(analyzer):
    # Test deduplication, stripping, and lowercasing
    input_text = "Rock, pop , ROCK, acoustic ,  jazz, Pop"
    result = analyzer._clean_tags(input_text)
    assert result == "rock, pop, acoustic, jazz"

    # Test limit to 20 unique tags
    many_tags = ", ".join([f"tag{i}" for i in range(30)])
    result = analyzer._clean_tags(many_tags)
    assert len(result.split(", ")) == 20
