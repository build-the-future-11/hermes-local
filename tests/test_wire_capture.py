"""Real client reads with instrumented, offline HTTPX byte streams."""

import json
from collections.abc import Iterable

import httpx
import pytest

from hermes_local import backend
from hermes_local.backend import OllamaClient
from hermes_local.chat import HermesChat


class ObservedStream(httpx.SyncByteStream):
    def __init__(self, chunks: Iterable[bytes]) -> None:
        self.chunks = chunks
        self.consumed = 0
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk

    def close(self) -> None:
        self.closed = True


def encoded(content: str = "answer", **overrides) -> bytes:
    return json.dumps({
        "model": "base", "message": {"content": content},
        "done": True, "done_reason": "stop", **overrides,
    }, ensure_ascii=False).encode("utf-8")


def client(stream: ObservedStream, **response_kwargs) -> OllamaClient:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream, **response_kwargs)

    return OllamaClient(transport=httpx.MockTransport(handle))


@pytest.mark.parametrize("streaming", [False, True])
def test_client_explicitly_requests_identity_encoding(streaming):
    requested = []

    def handle(request):
        requested.append(request.headers["accept-encoding"])
        return httpx.Response(200, stream=ObservedStream([encoded()]))

    provider = OllamaClient(transport=httpx.MockTransport(handle))
    provider.generate(model="base", prompt="hi", on_token=(lambda _: None) if streaming else None)
    assert requested == ["identity"]


@pytest.mark.parametrize("streaming", [False, True])
def test_complete_response_limit_accepts_boundary_and_refuses_next_byte(monkeypatch, streaming):
    raw = encoded()
    monkeypatch.setattr(backend, "MAX_RESPONSE_BYTES", len(raw))
    events = []
    valid = ObservedStream([raw[:5], raw[5:]])
    assert client(valid).generate(
        model="base", prompt="hi", on_token=events.append if streaming else None,
    ).content == "answer"
    assert valid.closed
    oversized = ObservedStream([raw + b" ", b"unread"])
    with pytest.raises(ValueError, match="wire limit"):
        client(oversized).generate(
            model="base", prompt="hi", on_token=events.append if streaming else None,
        )
    assert oversized.consumed == 1
    assert oversized.closed


def test_nonstream_ignored_metadata_cannot_bypass_wire_bound(monkeypatch):
    raw = encoded(padding="x" * 1000)
    monkeypatch.setattr(backend, "MAX_RESPONSE_BYTES", 128)
    stream = ObservedStream([raw[:100], raw[100:200], raw[200:]])
    with pytest.raises(ValueError, match="wire limit"):
        client(stream).generate(model="base", prompt="hi")
    assert stream.consumed == 2
    assert stream.closed


def test_unterminated_record_is_bounded_before_waiting_for_newline(monkeypatch):
    monkeypatch.setattr(backend, "MAX_STREAM_RECORD_BYTES", 128)
    stream = ObservedStream([b" " * 64, b" " * 65, encoded()])
    emitted = []
    with pytest.raises(ValueError, match="record exceeds"):
        client(stream).generate(model="base", prompt="hi", on_token=emitted.append)
    assert stream.consumed == 2
    assert stream.closed
    assert emitted == []


def test_record_bound_accepts_exact_size_and_counts_cr_byte(monkeypatch):
    raw = encoded()
    monkeypatch.setattr(backend, "MAX_STREAM_RECORD_BYTES", len(raw))
    stream = ObservedStream([raw + b"\n"])
    result = client(stream).generate(model="base", prompt="hi", on_token=lambda _: None)
    assert result.content == "answer"
    stream = ObservedStream([raw + b"\r\n"])
    with pytest.raises(ValueError, match="record exceeds"):
        client(stream).generate(model="base", prompt="hi", on_token=lambda _: None)
    assert stream.closed


def test_blank_lines_count_toward_record_limit(monkeypatch):
    monkeypatch.setattr(backend, "MAX_STREAM_RECORDS", 4)
    stream = ObservedStream([b"\n" * 5, encoded()])
    emitted = []
    with pytest.raises(ValueError, match="record limit"):
        client(stream).generate(model="base", prompt="hi", on_token=emitted.append)
    assert stream.consumed == 1
    assert stream.closed
    assert emitted == []


def test_final_unterminated_record_counts_toward_limit(monkeypatch):
    monkeypatch.setattr(backend, "MAX_STREAM_RECORDS", 2)
    at_limit = ObservedStream([b"\n", encoded()])
    result = client(at_limit).generate(model="base", prompt="hi", on_token=lambda _: None)
    assert result.content == "answer"
    excess = ObservedStream([b"\n\n", encoded()])
    with pytest.raises(ValueError, match="record limit"):
        client(excess).generate(model="base", prompt="hi", on_token=lambda _: None)
    assert excess.closed


@pytest.mark.parametrize("split", [1, 2, 3, 7, 64, 1000])
def test_chunk_boundaries_utf8_crlf_and_final_record_are_lossless(split):
    first = encoded("café 🛰️", done=False, done_reason=None)
    second = encoded(" complete")
    raw = b"\r\n" + first + b"\r\n" + second
    stream = ObservedStream(raw[i:i + split] for i in range(0, len(raw), split))
    emitted = []
    result = client(stream).generate(model="base", prompt="hi", on_token=emitted.append)
    assert result.content == "café 🛰️ complete"
    assert emitted == ["café 🛰️", " complete"]
    assert stream.closed


@pytest.mark.parametrize("raw", [
    b'{"model":"other","model":"base","message":{"content":"bad"},"done":true,"done_reason":"stop"}',
    b'{"model":"base","message":{"content":"bad"},"done":false,"done":true,"done_reason":"stop"}',
    b'{"model":"base","message":{"content":"first","content":"bad"},"done":true,"done_reason":"stop"}',
])
@pytest.mark.parametrize("streaming", [False, True])
def test_duplicate_identity_or_content_is_refused_before_emission(raw, streaming):
    stream = ObservedStream([raw])
    emitted = []
    with pytest.raises(ValueError, match="duplicate JSON"):
        client(stream).generate(
            model="base", prompt="hi", on_token=emitted.append if streaming else None,
        )
    assert emitted == []
    assert stream.closed


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_nonfinite_ignored_provider_metadata_is_rejected(token):
    raw = encoded()[:-1] + b',"metadata":[' + token.encode() + b"]}"
    stream = ObservedStream([raw])
    with pytest.raises(ValueError, match="JSON number"):
        client(stream).generate(model="base", prompt="hi")
    assert stream.closed


@pytest.mark.parametrize("streaming", [False, True])
def test_invalid_utf8_is_rejected_without_replacement_character(streaming):
    raw = encoded().replace(b"answer", b"bad\xffanswer")
    stream = ObservedStream([raw])
    emitted = []
    with pytest.raises(ValueError, match="UTF-8"):
        client(stream).generate(
            model="base", prompt="hi", on_token=emitted.append if streaming else None,
        )
    assert emitted == []
    assert stream.closed


@pytest.mark.parametrize("encoding", ["gzip", "deflate", "br", "identity, gzip"])
@pytest.mark.parametrize("streaming", [False, True])
def test_encoded_body_is_refused_before_transport_iteration(encoding, streaming):
    stream = ObservedStream([encoded()])
    with pytest.raises(ValueError, match="identity content encoding"):
        client(stream, headers={"content-encoding": encoding}).generate(
            model="base", prompt="hi", on_token=(lambda _: None) if streaming else None,
        )
    assert stream.consumed == 0
    assert stream.closed


def test_failed_terminal_json_keeps_history_and_closes_response():
    streams = []
    chat_calls = 0

    def handle(request):
        nonlocal chat_calls
        if request.url.path == "/api/tags":
            raw = json.dumps({"models": [{"name": "base", "digest": "a" * 64}]}).encode()
        else:
            chat_calls += 1
            raw = encoded() if chat_calls == 1 else (
                encoded("partial", done=False, done_reason=None) + b"\n" +
                encoded().replace(b'"done": true', b'"done": false, "done": true')
            )
        stream = ObservedStream([raw])
        streams.append(stream)
        return httpx.Response(200, stream=stream)

    session = HermesChat(OllamaClient(transport=httpx.MockTransport(handle)), "base")
    session.answer("first")
    previous = [dict(turn) for turn in session.history]
    emitted = []
    with pytest.raises(ValueError, match="duplicate JSON"):
        session.answer("second", on_token=emitted.append)
    assert emitted == ["partial"]
    assert session.history == previous
    assert all(stream.closed for stream in streams)


def test_callback_exception_closes_transport_without_reading_later_data():
    stream = ObservedStream([encoded("partial", done=False, done_reason=None) + b"\n", encoded()])

    def interrupted(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        client(stream).generate(model="base", prompt="hi", on_token=interrupted)
    assert stream.closed
    assert stream.consumed == 1


def test_version_and_tags_share_the_bounded_decoder(monkeypatch):
    monkeypatch.setattr(backend, "MAX_RESPONSE_BYTES", 64)
    for method in ("version", "list_models"):
        stream = ObservedStream([b" " * 65, b'{}'])
        with pytest.raises(ValueError, match="wire limit"):
            getattr(client(stream), method)()
        assert stream.consumed == 1
        assert stream.closed
