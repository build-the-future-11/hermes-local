import json

import httpx
import pytest

from hermes_local.backend import OllamaClient
from hermes_local.chat import HermesChat


def test_chat_retains_turns_and_clears_and_reports_length() -> None:
    requests: list[dict[str, object]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "base", "digest": "a" * 64}]})
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"model": "base", "message": {"content": "hello"},
                                        "done": True, "done_reason": "length"})

    session = HermesChat(OllamaClient(transport=httpx.MockTransport(handle)), "base")
    assert session.answer("first").finish_reason == "length"
    session.answer("second")
    assert requests[1]["messages"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "second"},
    ]
    session.clear()
    assert session.history == []


def test_chat_bounds_history_and_rejects_changed_model() -> None:
    digest = "a" * 64

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "base", "digest": digest}]})
        return httpx.Response(200, json={"model": "base", "message": {"content": "x" * 80},
                                        "done": True, "done_reason": "stop"})

    session = HermesChat(OllamaClient(transport=httpx.MockTransport(handle)), "base",
                         context_tokens=256, max_tokens=16)
    session.answer("hello")
    session.answer("again")
    assert session.truncated
    assert len(session.history) == 2
    before = list(session.history)
    with pytest.raises(ValueError, match="budget"):
        session.answer("x" * 500)
    digest = "b" * 64
    with pytest.raises(ValueError, match="digest changed"):
        session.answer("hello")
    assert session.history == before
