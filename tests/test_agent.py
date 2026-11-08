"""Harness behaviour with a scripted fake model: no API key or network needed."""
import json
from types import SimpleNamespace

import pytest

from motor_rag import agent
from motor_rag.agent import Harness
from motor_rag.ingest import Chunk


def chunk(n, insurer="SBI General", section="7. DEDUCTIBLE"):
    return Chunk(chunk_id=f"d#{n}", doc_id="sbi_pc_saod_pw", insurer=insurer, product="Own Damage",
                 vehicle="private_car", doc_type="wording", page=1, section=section, text=f"clause text {n}")


class FakeRetriever:
    def __init__(self):
        self.chunks = [chunk(1), chunk(2), chunk(3, section="6. GENERAL EXCEPTIONS")]
        self.calls = 0

    def search(self, query, k=5, mode="hybrid_rerank", insurer=None, vehicle=None):
        self.calls += 1
        return self.chunks[:2]

    def rerank(self, query, chunks, k):
        return chunks[:k]


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(agent, "load_sources", lambda: [
        {"insurer": "SBI General", "product": "Own Damage", "vehicle": "private_car"}])
    return Harness(FakeRetriever(), use_graph=False)


@pytest.fixture
def graph_harness(monkeypatch):
    monkeypatch.setattr(agent, "load_sources", lambda: [
        {"insurer": "SBI General", "product": "Own Damage", "vehicle": "private_car"}])
    calls = []

    def fake_concept_by_insurer(concept, insurers, vehicle=None):
        calls.append((concept, tuple(insurers), vehicle))
        return {insurers[0]: ["d#1", "d#3"]} if vehicle != "goods_carrying" else {}
    monkeypatch.setattr(agent.graph, "concept_by_insurer", fake_concept_by_insurer)
    monkeypatch.setattr(agent.graph, "related_clauses", lambda chunk_id: ["d#3"])
    h = Harness(FakeRetriever(), use_graph=True)
    h.graph_calls = calls
    return h


def test_graph_tools_are_only_offered_when_the_graph_is_available(harness, graph_harness):
    names = lambda h: {t["function"]["name"] for t in h.tools}
    assert "compare_clauses" not in names(harness)
    assert harness.call("compare_clauses", json.dumps({"question": "q", "concept": "towing", "insurers": ["SBI General"]}))[1] == "unknown tool"
    assert {"compare_clauses", "follow_references"} <= names(graph_harness)


def test_compare_clauses_runs_one_graph_query_per_insurer_and_vehicle(graph_harness):
    result, note = graph_harness.call("compare_clauses", json.dumps({
        "question": "towing limits", "concept": "towing", "insurers": ["SBI General"],
        "vehicles": ["private_car", "goods_carrying"]}))
    assert note == ""
    assert "=== SBI General / private_car ===" in result and "=== SBI General / goods_carrying ===" in result
    assert "No clauses about towing." in result  # the fake graph has none for goods vehicles
    assert [c[2] for c in graph_harness.graph_calls] == ["private_car", "goods_carrying"]


def test_compare_clauses_rejects_a_one_sided_comparison_and_unknown_concepts(graph_harness):
    one_side = {"question": "q", "concept": "towing", "insurers": ["SBI General"]}
    assert "two or more" in graph_harness.call("compare_clauses", json.dumps(one_side))[0]
    bad = {"question": "q", "concept": "spaceships", "insurers": ["SBI General", "SBI General"]}
    assert graph_harness.call("compare_clauses", json.dumps(bad))[1] == "tool error"


def test_follow_references_checks_the_excerpt_number(graph_harness):
    assert graph_harness.call("follow_references", json.dumps({"excerpt": 1}))[1] == "tool error"  # nothing numbered yet
    graph_harness.call("search_policy_clauses", json.dumps({"query": "deductible"}))
    result, _ = graph_harness.call("follow_references", json.dumps({"excerpt": 1}))
    assert result.startswith("[3]")


def test_unknown_insurer_returns_an_error_listing_valid_ones(harness):
    result, note = harness.call("search_policy_clauses", json.dumps({"query": "towing", "insurer": "Acme"}))
    assert result.startswith("Error: unknown insurer") and "SBI General" in result
    assert note == "tool error"


def test_insurer_name_is_matched_case_insensitively(harness):
    result, _ = harness.call("search_policy_clauses", json.dumps({"query": "towing", "insurer": "sbi general"}))
    assert result.startswith("[1]")


def test_bad_json_and_unknown_tools_do_not_crash(harness):
    assert harness.call("search_policy_clauses", "{not json")[1] == "bad arguments"
    assert harness.call("search_policy_clauses", json.dumps({"q": "x"}))[1] == "bad arguments"
    assert harness.call("delete_everything", "{}")[1] == "unknown tool"


def test_repeated_identical_call_is_served_from_cache(harness):
    args = json.dumps({"query": "towing"})
    harness.call("search_policy_clauses", args)
    result, note = harness.call("search_policy_clauses", args)
    assert note == "repeated call" and harness.retriever.calls == 1


def test_excerpt_numbers_are_global_across_tool_calls(harness):
    harness.call("search_policy_clauses", json.dumps({"query": "a"}))  # returns chunks 1, 2
    result, _ = harness.call("get_section", json.dumps({"doc_id": "sbi_pc_saod_pw", "section": "6. GENERAL EXCEPTIONS"}))
    assert result.startswith("[3]")  # a new chunk gets the next number, not [1] again
    assert len(harness.excerpts) == 3


def test_already_retrieved_chunk_is_deduplicated_in_subsequent_call(harness):
    harness.call("search_policy_clauses", json.dumps({"query": "a"}))  # returns chunks 1, 2
    # Second search with different query returns fake chunks 1 and 2 again:
    result, _ = harness.call("search_policy_clauses", json.dumps({"query": "b"}))
    assert "(Already retrieved above; refer to excerpt [1])" in result


def tool_call(name, args, id="call_1"):
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    return SimpleNamespace(id=id, function=fn, model_dump=lambda: {"id": id, "type": "function",
                                                                   "function": {"name": name, "arguments": json.dumps(args)}})


def scripted_chat(replies):
    """Fake llm.chat that returns the given messages in order."""
    replies = iter(replies)

    def fake_chat(messages, **kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=next(replies))])
    return fake_chat


def test_run_searches_then_answers_with_citations(monkeypatch):
    monkeypatch.setattr(agent, "load_sources", lambda: [{"insurer": "SBI General", "product": "OD", "vehicle": "private_car"}])
    monkeypatch.setattr(agent, "chat", scripted_chat([
        SimpleNamespace(content=None, tool_calls=[tool_call("search_policy_clauses", {"query": "deductible"})]),
        SimpleNamespace(content="The deductible is in clause [2].", tool_calls=None),
    ]))
    result = agent.run("What is the deductible?", FakeRetriever(), use_graph=False)
    assert result.text == "The deductible is in clause [2]."
    assert [c.chunk_id for c in result.cited] == ["d#2"]
    assert result.steps == 1 and not result.stopped_by_limit


def test_run_stops_at_the_step_limit_and_still_answers(monkeypatch):
    monkeypatch.setattr(agent, "load_sources", lambda: [{"insurer": "SBI General", "product": "OD", "vehicle": "private_car"}])
    looping = [SimpleNamespace(content=None, tool_calls=[tool_call("search_policy_clauses", {"query": f"q{i}"}, id=f"c{i}")])
               for i in range(3)]
    monkeypatch.setattr(agent, "chat", scripted_chat(looping + [SimpleNamespace(content="Best effort [1].", tool_calls=None)]))
    result = agent.run("question", FakeRetriever(), max_steps=3, use_graph=False)
    assert result.stopped_by_limit and result.steps == 3
    assert result.text == "Best effort [1]."
