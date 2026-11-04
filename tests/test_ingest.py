import pymupdf

from motor_rag.ingest import MAX_CHARS, chunk_document, clean, is_heading, page_blocks, split_long


def test_real_section_headings_are_detected():
    assert is_heading("6. GENERAL EXCEPTIONS")
    assert is_heading("SECTION II - LIABILITY TO THIRD PARTIES")
    assert is_heading("EXCLUSIONS")
    assert is_heading("ZERO DEPRECIATION COVER")


def test_table_cells_and_sentences_are_not_headings():
    # City names from a branch-office table produced hundreds of junk sections before.
    assert not is_heading("HYDERABAD")
    assert not is_heading("SECUNDERABAD")
    assert not is_heading("The Company will indemnify the insured against loss")
    assert not is_heading("5%")


def test_clean_fixes_hyphenation_ligatures_and_missing_dashes():
    assert clean("depre-\nciation") == "depreciation"
    assert clean("ﬁre  and\nﬂood") == "fire and flood"
    assert clean("SECTION I � LOSS") == "SECTION I - LOSS"


def test_split_long_keeps_short_text_whole_and_splits_long_text_at_sentences():
    assert split_long("Short clause.") == ["Short clause."]
    text = " ".join(f"Sentence number {i} about depreciation." for i in range(200))
    pieces = split_long(text)
    assert len(pieces) > 1
    assert all(len(p) <= MAX_CHARS * 1.5 for p in pieces)
    assert all(p.endswith(".") for p in pieces)


def two_column_pdf(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=600, height=800)
    page.insert_text((40, 60), "LEFT TOP")
    page.insert_text((40, 400), "LEFT BOTTOM")
    page.insert_text((340, 60), "RIGHT TOP")
    page.insert_text((340, 400), "RIGHT BOTTOM")
    path = tmp_path / "two_col.pdf"
    doc.save(path)
    return path


def test_two_column_pages_are_read_column_by_column(tmp_path):
    with pymupdf.open(two_column_pdf(tmp_path)) as doc:
        order = [b.strip() for b in page_blocks(doc[0])]
    # Reading across would give LEFT TOP, RIGHT TOP, ...
    assert order == ["LEFT TOP", "LEFT BOTTOM", "RIGHT TOP", "RIGHT BOTTOM"]


META = {"doc_id": "test_doc", "insurer": "Test General", "product": "Private Car",
        "vehicle": "private_car", "doc_type": "wording"}


def test_heading_split_over_two_lines_is_joined(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((40, 60), "7.\nDEDUCTIBLE")
    page.insert_textbox(pymupdf.Rect(40, 100, 560, 700), "The insured shall bear the first Rs. 1000 of each claim. " * 6)
    path = tmp_path / "split.pdf"
    doc.save(path)
    assert chunk_document(path, META)[0].section == "7. DEDUCTIBLE"


def test_runs_of_table_cells_do_not_become_sections(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 560, 200), "Roadside assistance is available in the following cities only. " * 4)
    for i, city in enumerate(["FATTEPUR SIKRI", "NIZAMABAD", "BISHNUPUR WEST"]):
        page.insert_text((40, 260 + i * 20), city)
    path = tmp_path / "cities.pdf"
    doc.save(path)
    sections = {c.section for c in chunk_document(path, META)}
    assert "BISHNUPUR WEST" not in sections


def test_chunks_carry_source_metadata(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((40, 60), "4. EXCLUSIONS")
    page.insert_textbox(pymupdf.Rect(40, 80, 560, 700), "The Company shall not be liable for consequential loss. " * 10)
    path = tmp_path / "policy.pdf"
    doc.save(path)
    meta = {"doc_id": "test_doc", "insurer": "Test General", "product": "Private Car",
            "vehicle": "private_car", "doc_type": "wording"}
    chunks = chunk_document(path, meta)
    assert chunks[0].section == "4. EXCLUSIONS"
    assert chunks[0].insurer == "Test General"
    assert chunks[0].page == 1
    assert "consequential loss" in chunks[0].text
