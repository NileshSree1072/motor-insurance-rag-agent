from motor_rag.ingest import Chunk
from motor_rag.rag import cited_chunks, tidy

CHUNKS = [Chunk(f"d#{i}", "d", "Ins", "P", "private_car", "wording", 1, "S", f"text {i}") for i in range(1, 6)]


def ids(text):
    return [c.chunk_id for c in cited_chunks(text, CHUNKS)]


def test_square_bracket_citations():
    assert ids("Towing is Rs 1500 [2]. Glass is nil [4][2].") == ["d#2", "d#4"]


def test_full_width_bracket_citations():
    # gpt-oss writes 【1】; missing these made citation accuracy look like 31%.
    assert ids("Up to two claims【1】 and four RSA claims【3】.") == ["d#1", "d#3"]


def test_comma_separated_citations_and_out_of_range_numbers():
    assert ids("Both apply [1, 3]. Unknown [9].") == ["d#1", "d#3"]


def test_tidy_normalises_spaces_and_brackets():
    assert tidy("50 % depreciation【2】 ") == "50 % depreciation[2]"
