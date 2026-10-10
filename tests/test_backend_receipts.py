"""Provider receipts must prove completion before an answer enters history."""

import json
from typing import Any

import httpx
import pytest

from hermes_local.backend import OllamaClient
from hermes_local.chat import HermesChat


def receipt(**overrides: Any) -> dict[str, Any]:
    return {
        "model": "base",
        "message": {"role": "assistant", "content": "complete answer"},
        "done": True,
        "done_reason": "stop",
        **overrides,
    }


def client_for(payload: dict[str, Any]) -> OllamaClient:
    return OllamaClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json=payload)))


@pytest.mark.parametrize("done", [False, None, 0, 1, "true"])
def test_nonstream_requires_explicit_completion(done: Any) -> None:
    with pytest.raises(ValueError, match="completion"):
        client_for(receipt(done=done)).generate(model="base", prompt="hello")


@pytest.mark.parametrize("reason", [None, "", "load", "unload", "unknown", 1, []])
def test_unknown_completion_reason_is_not_inferred_as_stop(reason: Any) -> None:
    with pytest.raises(ValueError, match="done_reason"):
        client_for(receipt(done_reason=reason)).generate(model="base", prompt="hello")


def test_missing_completion_reason_is_rejected() -> None:
    payload = receipt()
    del payload["done_reason"]
    with pytest.raises(ValueError, match="done_reason"):
        client_for(payload).generate(model="base", prompt="hello")


@pytest.mark.parametrize("reason", ["stop", "length"])
def test_supported_receipt_preserves_provider_finish_reason(reason: str) -> None:
    result = client_for(receipt(done_reason=reason, eval_count=2)).generate(
        model="base", prompt="hello")
    assert result.finish_reason == reason
    assert result.evidence["done_reason"] == reason
    assert result.evidence["eval_count"] == 2
    assert result.content == "complete answer"


@pytest.mark.parametrize("role", ["user", "tool", None, 1])
def test_nonassistant_message_is_not_an_answer(role: Any) -> None:
    payload = receipt(message={"role": role, "content": "wrong role"})
    with pytest.raises(ValueError, match="malformed"):
        client_for(payload).generate(model="base", prompt="hello")


def test_error_field_cannot_coexist_with_a_successful_receipt() -> None:
    with pytest.raises(ValueError, match="malformed"):
        client_for(receipt(error="out of memory")).generate(model="base", prompt="hello")


def test_nonstream_output_limit_counts_utf8_bytes() -> None:
    at_limit = "é" * (1_048_576 // 2)
    assert client_for(receipt(message={"content": at_limit})).generate(
        model="base", prompt="hello").content == at_limit
    with pytest.raises(ValueError, match="output limit"):
        client_for(receipt(message={"content": at_limit + "é"})).generate(
            model="base", prompt="hello")


@pytest.mark.parametrize("terminal_changes", [
    {"done_reason": None},
    {"done_reason": "unknown"},
    {"total_duration": -1},
    {"eval_count": True},
    {"load_duration": 1.5},
    {"message": {"role": "user", "content": "invalid final text"}},
])
def test_invalid_terminal_stream_event_is_not_emitted_or_saved(
    terminal_changes: dict[str, Any],
) -> None:
    events = [receipt(done=False, done_reason=None,
                      message={"content": "provisional text"}),
              receipt(**{**{"message": {"content": "invalid final text"}},
                         **terminal_changes})]

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "base", "digest": "a" * 64}]})
        return httpx.Response(200, text="\n".join(json.dumps(event) for event in events))

    session = HermesChat(OllamaClient(transport=httpx.MockTransport(handle)), "base")
    emitted: list[str] = []
    with pytest.raises(ValueError):
        session.answer("hello", on_token=emitted.append)
    assert emitted == ["provisional text"]
    assert session.history == []


def test_nonterminal_event_must_not_claim_a_completion_reason() -> None:
    emitted: list[str] = []
    client = OllamaClient(transport=httpx.MockTransport(lambda request: httpx.Response(
        200, text=json.dumps(receipt(done=False)))))
    with pytest.raises(ValueError, match="completion reason"):
        client.generate(model="base", prompt="hello", on_token=emitted.append)
    assert emitted == []


@pytest.mark.parametrize("field,value", [
    ("max_tokens", True), ("max_tokens", 3.5),
    ("context_tokens", 512.5), ("seed", True), ("seed", 1.5),
    ("temperature", True), ("temperature", "0.5"),
    ("system", []), ("model", 7), ("prompt", 7), ("history", [None, None]),
])
def test_malformed_parameters_fail_before_http(field: str, value: Any) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=receipt())

    client = OllamaClient(transport=httpx.MockTransport(handle))
    with pytest.raises(ValueError):
        client.generate(**{"model": "base", "prompt": "hello", field: value})
    assert requests == []


@pytest.mark.parametrize("timeout", [True, "120", float("inf"), float("nan"), 0])
def test_invalid_timeout_has_a_consistent_validation_error(timeout: Any) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        OllamaClient(timeout_seconds=timeout)


def test_ambiguous_installed_model_identity_is_rejected() -> None:
    models = [{"name": "base", "digest": "a" * 64},
              {"name": "base", "digest": "b" * 64}]
    client = client_for({"models": models})
    with pytest.raises(ValueError, match="duplicate"):
        client.model_digest("base")


@pytest.mark.parametrize("prefix", ["", "sha256:"])
def test_unambiguous_digest_is_normalized(prefix: str) -> None:
    client = client_for({"models": [{"name": "base", "digest": prefix + "a" * 64}]})
    assert client.model_digest("base") == "sha256:" + "a" * 64


def test_incomplete_nonstream_response_preserves_existing_session_history() -> None:
    chat_requests = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal chat_requests
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "base", "digest": "a" * 64}]})
        chat_requests += 1
        return httpx.Response(200, json=receipt(done=chat_requests == 1))

    session = HermesChat(OllamaClient(transport=httpx.MockTransport(handle)), "base")
    session.answer("first")
    before = [dict(turn) for turn in session.history]
    with pytest.raises(ValueError, match="completion"):
        session.answer("second")
    assert session.history == before
