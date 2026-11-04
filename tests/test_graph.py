"""Graph extraction rules (no Neo4j needed)."""
from motor_rag.graph import chunk_rows, concepts_in, definitions_in, normalize_ref, references_in
from motor_rag.ingest import Chunk


def test_concepts_match_policy_wording_not_just_everyday_words():
    text = "the Company will bear the reasonable cost of protection and removal to the nearest repairer"
    assert "towing" in concepts_in(text)
    assert "drunk driving" in concepts_in("whilst under the influence of intoxicating liquor or drugs")


def test_battery_alone_is_not_an_electric_vehicle_clause():
    # Every depreciation table lists batteries; that used to tag 498 chunks as EV.
    assert "electric vehicle" not in concepts_in("rubber, nylon, plastic parts, tyres, tubes and batteries - 50%")


def test_references_are_normalised():
    assert normalize_ref("IMT. 22") == "IMT 22"
    assert normalize_ref("Section 2") == "SECTION II"
    assert references_in("subject to IMT. 22 and Section ii of the policy") == ["IMT 22", "SECTION II"]


def test_acts_and_large_section_numbers_are_not_policy_references():
    assert references_in("as defined in Section 56 of the Motor Vehicles Act") == []


def test_definitions_need_an_upper_case_title():
    assert definitions_in("16. IMT. 22. COMPULSORY DEDUCTIBLE (Applicable to private cars)") == ["IMT 22"]
    assert definitions_in("payable subject to IMT 22 of the tariff") == []


def chunk(n, section, text):
    return Chunk(f"d#{n}", "d", "Ins", "P", "private_car", "wording", 1, section, text)


def test_only_the_first_chunk_of_a_section_defines_its_heading():
    rows = chunk_rows([
        chunk(1, "SECTION II - LIABILITY TO THIRD PARTIES", "The Company will indemnify the insured"),
        chunk(2, "SECTION II - LIABILITY TO THIRD PARTIES", "against all sums which the insured shall become liable"),
    ])
    assert rows[0]["defines"] == ["SECTION II"]
    assert rows[1]["defines"] == []
    assert rows[1]["prev"] == "d#1"


def test_a_chunk_does_not_refer_to_the_clause_it_defines():
    rows = chunk_rows([chunk(1, "ENDORSEMENTS", "IMT. 22. COMPULSORY DEDUCTIBLE The insured shall bear under IMT 22")])
    assert rows[0]["defines"] == ["IMT 22"] and rows[0]["refs"] == []
