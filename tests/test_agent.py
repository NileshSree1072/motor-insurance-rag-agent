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


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(agent, "load_sources", lambda: [
        {"insurer": "SBI General", "product": "Own Damage", "vehicle": "private_car"}])
    return Harness(FakeRetriever())


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
    result = agent.run("What is the deductible?", FakeRetriever())
    assert result.text == "The deductible is in clause [2]."
    assert [c.chunk_id for c in result.cited] == ["d#2"]
    assert result.steps == 1 and not result.stopped_by_limit


def test_run_stops_at_the_step_limit_and_still_answers(monkeypatch):
    monkeypatch.setattr(agent, "load_sources", lambda: [{"insurer": "SBI General", "product": "OD", "vehicle": "private_car"}])
    looping = [SimpleNamespace(content=None, tool_calls=[tool_call("search_policy_clauses", {"query": f"q{i}"}, id=f"c{i}")])
               for i in range(3)]
    monkeypatch.setattr(agent, "chat", scripted_chat(looping + [SimpleNamespace(content="Best effort [1].", tool_calls=None)]))
    result = agent.run("question", FakeRetriever(), max_steps=3)
    assert result.stopped_by_limit and result.steps == 3
    assert result.text == "Best effort [1]."
