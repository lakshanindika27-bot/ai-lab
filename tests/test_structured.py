import importlib.util
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

SRC = Path(__file__).resolve().parents[1] / "02-structured" / "ask_structured.py"
spec = importlib.util.spec_from_file_location("ask_structured", SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

CHUNK = {
    "source": "git_basics.txt",
    "text": "Run git init to create a new repository.",
    "distance": 0.3,
}
GOOD = {
    "answer": "Run git init",
    "answerable": True,
    "citations": ["git_basics.txt"],
    "evidence": "Run git init to create a new repository.",
}


@pytest.fixture
def llm(monkeypatch):
    replies = []
    calls = {"n": 0}

    def fake_chat(**kwargs):
        i = min(calls["n"], len(replies) - 1)
        calls["n"] += 1
        return {"message": {"content": replies[i]}}

    monkeypatch.setattr(mod, "retrieve", lambda q, k: [CHUNK])
    monkeypatch.setattr(mod.ollama, "chat", fake_chat)
    monkeypatch.setattr(mod, "GROUNDED", False)
    monkeypatch.setattr(mod, "VERIFY", False)
    return replies, calls


def test_answerable_requires_citations_and_evidence():
    with pytest.raises(ValidationError):
        mod.Answer(answer="x", answerable=True)


def test_unanswerable_needs_no_citations():
    a = mod.Answer(answer="no", answerable=False)
    assert a.citations == [] and a.evidence == ""


def test_distance_gate_skips_llm(monkeypatch):
    far = dict(CHUNK, distance=0.99)
    monkeypatch.setattr(mod, "retrieve", lambda q, k: [far])

    def boom(**kw):
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(mod.ollama, "chat", boom)
    assert mod.ask("q") == mod.NO_ANSWER


def test_valid_answer_returned_first_try(llm):
    replies, calls = llm
    replies.append(json.dumps(GOOD))
    result = mod.ask("how to start?")
    assert result.answer == "Run git init"
    assert calls["n"] == 1


def test_invalid_json_retries_then_succeeds(llm):
    replies, calls = llm
    replies.extend(["not json", json.dumps(GOOD)])
    assert mod.ask("q").answerable is True
    assert calls["n"] == 2


def test_hallucinated_citation_is_rejected(llm):
    replies, calls = llm
    bad = dict(GOOD, citations=["made_up.txt"])
    replies.append(json.dumps(bad))
    assert mod.ask("q") == mod.NO_ANSWER
    assert calls["n"] == 2  # identical reply on retry -> early exit


def test_fake_evidence_is_rejected(llm):
    replies, _ = llm
    bad = dict(GOOD, evidence="Something the context never said.")
    replies.append(json.dumps(bad))
    assert mod.ask("q") == mod.NO_ANSWER


def test_model_abstains(llm):
    replies, calls = llm
    replies.append(json.dumps({"answer": "I do not know", "answerable": False}))
    assert mod.ask("q") == mod.NO_ANSWER
    assert calls["n"] == 1


def test_verifier_rejection_forces_retries(llm, monkeypatch):
    replies, calls = llm
    replies.append(json.dumps(GOOD))
    monkeypatch.setattr(mod, "VERIFY", True)
    monkeypatch.setattr(mod, "supports", lambda q, e, a: False)
    assert mod.ask("q") == mod.NO_ANSWER
    assert calls["n"] == 2  # identical reply on retry -> early exit


@pytest.mark.parametrize(
    "answer,evidence,expected",
    [
        ("git init", "Run git init to create a repo.", True),
        ("docker compose", "Run git init to create a repo.", False),
        ("the a an", "anything", False),
    ],
)
def test_grounded(answer, evidence, expected):
    assert mod.grounded(answer, evidence) is expected


def test_distinct_bad_replies_use_all_retries(llm):
    replies, calls = llm
    for i in range(mod.MAX_RETRIES + 1):
        replies.append(json.dumps(dict(GOOD, citations=[f"made_up_{i}.txt"])))
    assert mod.ask("q") == mod.NO_ANSWER
    assert calls["n"] == mod.MAX_RETRIES + 1
