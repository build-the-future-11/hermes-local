import json

import httpx
import pytest

from hermes_local.backend import OllamaClient
from hermes_local.chat import HermesChat


@pytest.mark.parametrize("complete", [True, False])
def test_streaming_commits_only_complete_turns(complete: bool) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "base", "digest": "a" * 64}]})
        assert json.loads(request.content)["stream"] is True
        events = [{"model": "base", "message": {"content": "Hello"}, "done": False}]
        if complete:
            events.append({"model": "base", "message": {"content": " world"}, "done": True})
        return httpx.Response(200, text="\n".join(json.dumps(event) for event in events))

    session = HermesChat(OllamaClient(transport=httpx.MockTransport(handle)), "base")
    tokens: list[str] = []
    if complete:
        result = session.answer("hi", on_token=tokens.append)
        assert result.content == "Hello world"
        assert tokens == ["Hello", " world"]
        assert len(session.history) == 2
    else:
        with pytest.raises(ValueError, match="without completion"):
            session.answer("hi", on_token=tokens.append)
        assert tokens == ["Hello"]
        assert session.history == []


def test_interrupt_does_not_commit_partial_turn() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "base", "digest": "a" * 64}]})
        return httpx.Response(200, text=json.dumps(
            {"model": "base", "message": {"content": "partial"}, "done": False}))

    def interrupt(text: str) -> None:
        raise KeyboardInterrupt

    session = HermesChat(OllamaClient(transport=httpx.MockTransport(handle)), "base")
    with pytest.raises(KeyboardInterrupt):
        session.answer("hello", on_token=interrupt)
    assert session.history == []
