import pytest
from nodes.lyrics_duration_node import AceStepLyricsBPMCalculator

@pytest.fixture
def calculator():
    return AceStepLyricsBPMCalculator()

def test_duration_and_counts(calculator):
    # 10 words, 2 lines
    lyrics = "This is a simple test line\nHere is another one"
    result = calculator.calculate(lyrics=lyrics, timesignature="4/4", wpm=120, line2bar=1.0)
    
    duration, bpm_low, bpm_mid, bpm_high, word_count, line_count = result
    
    assert word_count == 10
    assert line_count == 2
    # duration = max(120.0, (10 / 120) * 60 + 5.0) = max(120.0, 5.0 + 5.0 = 10.0) = 120.0
    assert duration == 120.0

def test_bpm_calculation(calculator):
    # Let's test non-capped duration to see BPM logic easily.
    # WPM=60. 120 words => (120/60)*60 = 120 seconds + 5 = 125 seconds.
    # 10 lines of 12 words
    lyrics = "\n".join(["One two three four five six seven eight nine ten eleven twelve" for _ in range(10)])
    
    result = calculator.calculate(lyrics=lyrics, timesignature="4/4", wpm=60, line2bar=1.0)
    duration, bpm_low, bpm_mid, bpm_high, word_count, line_count = result
    
    assert duration == 125.0
    assert line_count == 10
    assert word_count == 120
    
    # calc_bpm(density=1.0) => bars = 10 * 1.0 = 10
    # bpm = (10 * 4 * 60) / 125.0 = 2400 / 125 = 19.2 -> 19
    assert bpm_low == 19
    # mid density = 1.5 => bars = 15 => bpm = (15 * 4 * 60) / 125.0 = 3600 / 125 = 28.8 -> int() truncates to 28
    assert bpm_mid == 28

def test_ignore_tags(calculator):
    lyrics = "[Verse 1]\nSimple word\n[Chorus]\nAnother word"
    result = calculator.calculate(lyrics=lyrics, timesignature="4/4", wpm=150, line2bar=1.0)
    
    # only 2 valid lines, 4 words total
    _, _, _, _, word_count, line_count = result
    assert line_count == 2
    assert word_count == 4
