from __future__ import annotations

import ipaddress
import json
import math
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import httpx

from hermes_local.schemas import GenerationResult

MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_STREAM_RECORD_BYTES = 8 * 1024 * 1024
MAX_STREAM_RECORDS = 16_384


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Ollama returned a duplicate JSON object key")
        result[key] = value
    return result


def _finite_json_number(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValueError("Ollama returned a nonfinite JSON number")
    return value


def _reject_json_constant(token: str) -> Any:
    raise ValueError("Ollama returned a nonstandard JSON number")


def _parse_provider_object(raw: bytes | bytearray) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique_json_object,
            parse_float=_finite_json_number, parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, RecursionError) as error:
        raise ValueError("Ollama returned malformed UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("Ollama returned a non-object response")
    return payload


def _bounded_response_bytes(response: httpx.Response) -> Iterator[bytes]:
    # Identity avoids HTTPX decompressing a large body before our byte budget.
    if response.headers.get("content-encoding", "identity").strip().lower() != "identity":
        raise ValueError("Ollama response must use identity content encoding")
    consumed = 0
    for chunk in response.iter_bytes():
        consumed += len(chunk)
        if consumed > MAX_RESPONSE_BYTES:
            raise ValueError("Ollama response exceeds the 16 MiB wire limit")
        yield chunk


def _stream_objects(response: httpx.Response) -> Iterator[dict[str, Any]]:
    record = bytearray()
    count = 0
    for chunk in _bounded_response_bytes(response):
        start = 0
        while start < len(chunk):
            end = chunk.find(b"\n", start)
            stop = len(chunk) if end < 0 else end
            if len(record) + stop - start > MAX_STREAM_RECORD_BYTES:
                raise ValueError("Ollama stream record exceeds the 8 MiB wire limit")
            record.extend(memoryview(chunk)[start:stop])
            if end < 0:
                break
            count += 1
            if count > MAX_STREAM_RECORDS:
                raise ValueError("Ollama stream exceeds its record limit")
            if record.strip(b" \t\r"):
                yield _parse_provider_object(record)
            record.clear()
            start = end + 1
    if record:
        count += 1
        if count > MAX_STREAM_RECORDS:
            raise ValueError("Ollama stream exceeds its record limit")
        if record.strip(b" \t\r"):
            yield _parse_provider_object(record)


@dataclass(slots=True)
class OllamaClient:
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 120.0
    transport: httpx.BaseTransport | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        try:
            parsed = urlsplit(self.base_url)
            _ = parsed.port
        except ValueError as error:
            raise ValueError("Ollama base_url must be an HTTP or HTTPS origin") from error
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Ollama base_url must be an HTTP or HTTPS origin")
        try:
            is_loopback = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            is_loopback = parsed.hostname.casefold().rstrip(".") == "localhost"
        if parsed.scheme == "http" and not is_loopback:
            raise ValueError("remote Ollama endpoints require HTTPS")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        with httpx.Client(
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            transport=self.transport,
            trust_env=False,
            headers={"Accept-Encoding": "identity"},
        ) as client:
            with client.stream(method, path, **kwargs) as response:
                response.raise_for_status()
                raw = bytearray()
                for chunk in _bounded_response_bytes(response):
                    raw.extend(chunk)
                return _parse_provider_object(raw)

    def list_models(self) -> list[str]:
        payload = self._request("GET", "/api/tags")
        return self._model_names(payload.get("models"))

    @staticmethod
    def _model_names(models: Any) -> list[str]:
        if not isinstance(models, list):
            raise ValueError("Ollama model list is malformed")
        names: list[str] = []
        for model in models:
            if (
                not isinstance(model, dict)
                or not isinstance(model.get("name"), str)
                or not model["name"]
                or model["name"].strip() != model["name"]
            ):
                raise ValueError("Ollama model entry is malformed")
            names.append(model["name"])
        if len(names) != len(set(names)):
            raise ValueError("Ollama model list contains duplicate names")
        return names

    def version(self) -> str:
        version = self._request("GET", "/api/version").get("version")
        if (
            not isinstance(version, str)
            or not version
            or version.strip() != version
            or len(version) > 100
        ):
            raise ValueError("Ollama version response is malformed")
        return version

    def model_digest(self, name: str) -> str:
        if not name or name.strip() != name:
            raise ValueError("Ollama model name is malformed")
        payload = self._request("GET", "/api/tags")
        models = payload.get("models")
        self._model_names(models)
        for model in models:
            if isinstance(model, dict) and model.get("name") == name:
                digest = model.get("digest")
                match = (
                    re.fullmatch(r"(?:sha256:)?([0-9a-f]{64})", digest)
                    if isinstance(digest, str)
                    else None
                )
                if match is not None:
                    return f"sha256:{match.group(1)}"
                raise ValueError(f"Ollama model {name} has a malformed digest")
        raise ValueError(f"Ollama model is not installed: {name}")

    @staticmethod
    def _optional_nonnegative_integer(payload: dict[str, Any], field: str) -> int | None:
        value = payload.get(field)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Ollama chat response has malformed {field}")
        return value

    def health(self) -> dict[str, object]:
        models = self.list_models()
        return {"status": "ok", "provider": "ollama", "models": models}

    @staticmethod
    def _completion_reason(payload: dict[str, Any]) -> str:
        if payload.get("done") is not True:
            raise ValueError("Ollama chat response ended without explicit completion")
        reason = payload.get("done_reason")
        if reason not in ("stop", "length"):
            raise ValueError("Ollama chat response has missing or unsupported done_reason")
        return reason

    @staticmethod
    def _bounded_integer(value: Any, name: str, minimum: int, maximum: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        system: str | None = None,
        max_tokens: int = 256,
        context_tokens: int = 2_048,
        temperature: float = 0.0,
        seed: int | None = None,
        history: list[dict[str, str]] | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> GenerationResult:
        if not isinstance(model, str) or not model.strip() or model != model.strip():
            raise ValueError("model is required")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt is required")
        if system is not None and not isinstance(system, str):
            raise ValueError("system must be text")
        self._bounded_integer(max_tokens, "max_tokens", 1, 4_096)
        self._bounded_integer(context_tokens, "context_tokens", 256, 131_072)
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ValueError("temperature must be between 0 and 2")
        if seed is not None:
            self._bounded_integer(seed, "seed", 0, 2**32 - 1)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            for index, turn in enumerate(history):
                expected = "user" if index % 2 == 0 else "assistant"
                if (
                    not isinstance(turn, dict)
                    or set(turn) != {"role", "content"}
                    or turn["role"] != expected
                ):
                    raise ValueError("history must contain alternating user/assistant turns")
                if not isinstance(turn["content"], str) or not turn["content"].strip():
                    raise ValueError("history contains empty content")
            if len(history) % 2:
                raise ValueError("history must contain complete turns")
            messages.extend(dict(turn) for turn in history)
        messages.append({"role": "user", "content": prompt})
        request_body = {
                "model": model,
                "messages": messages,
                "stream": on_token is not None,
                "think": False,
                "options": {
                    "num_predict": max_tokens,
                    "num_ctx": context_tokens,
                    "temperature": temperature,
                    **({"seed": seed} if seed is not None else {}),
                },
            }
        if on_token is None:
            payload = self._request("POST", "/api/chat", json=request_body)
        else:
            payload = self._stream_chat(request_body, model, on_token)
        message = payload.get("message")
        if (
            "error" in payload
            or not isinstance(message, dict)
            or not isinstance(message.get("content"), str)
            or message.get("role", "assistant") != "assistant"
        ):
            raise ValueError("Ollama chat response is malformed")
        provider_model = payload.get("model")
        if provider_model != model:
            raise ValueError("Ollama chat response model does not match the request")
        done_reason = self._completion_reason(payload)
        total_duration = self._optional_nonnegative_integer(payload, "total_duration")
        eval_count = self._optional_nonnegative_integer(payload, "eval_count")
        load_duration = self._optional_nonnegative_integer(payload, "load_duration")
        content = message["content"]
        if len(content.encode("utf-8")) > 1_048_576:
            raise ValueError("generation response exceeds output limit")
        return GenerationResult(
            model=model,
            content=content,
            finish_reason=done_reason,
            prompt_characters=len(prompt),
            generated_characters=len(content),
            evidence={
                "provider": "ollama",
                "provider_model": provider_model,
                "done_reason": done_reason,
                "total_duration_ns": total_duration,
                "eval_count": eval_count,
                "load_duration_ns": load_duration,
            },
        )

    def _stream_chat(
        self, body: dict[str, Any], model: str, on_token: Callable[[str], None],
    ) -> dict[str, Any]:
        chunks: list[str] = []
        size = 0
        with httpx.Client(base_url=self.base_url, timeout=self.timeout_seconds,
                          transport=self.transport, trust_env=False,
                          headers={"Accept-Encoding": "identity"}) as client:
            with client.stream("POST", "/api/chat", json=body) as response:
                response.raise_for_status()
                for payload in _stream_objects(response):
                    if payload.get("model") != model:
                        raise ValueError("stream model identity is missing or mismatched")
                    message = payload.get("message")
                    if ("error" in payload or not isinstance(message, dict)
                            or not isinstance(message.get("content"), str)
                            or message.get("role", "assistant") != "assistant"
                            or not isinstance(payload.get("done"), bool)):
                        raise ValueError("malformed generation stream")
                    if payload["done"]:
                        self._completion_reason(payload)
                        for field in ("total_duration", "eval_count", "load_duration"):
                            self._optional_nonnegative_integer(payload, field)
                    elif payload.get("done_reason") not in (None, ""):
                        raise ValueError("unfinished stream event includes a completion reason")
                    content = message["content"]
                    size += len(content.encode("utf-8"))
                    if size > 1_048_576:
                        raise ValueError("generation stream exceeds output limit")
                    chunks.append(content)
                    if content:
                        on_token(content)
                    if payload["done"]:
                        payload["message"] = {"content": "".join(chunks)}
                        return payload
        raise ValueError("generation stream ended without completion")
