"""Explicit-base, ephemeral Hermes chat; not a qualified Hermes checkpoint."""

from collections.abc import Callable

from hermes_local.backend import OllamaClient
from hermes_local.schemas import GenerationResult


class HermesChat:
    def __init__(self, client: OllamaClient, model: str, *, context_tokens: int = 2048,
                 max_tokens: int = 256) -> None:
        if not 256 <= context_tokens <= 131072 or not 1 <= max_tokens <= 4096:
            raise ValueError("invalid context or output budget")
        if max_tokens + 128 >= context_tokens:
            raise ValueError("context must leave room for input and output")
        self.client = client
        self.model = model
        self.digest = client.model_digest(model)
        self.context_tokens = context_tokens
        self.max_tokens = max_tokens
        self.history: list[dict[str, str]] = []
        self.truncated = False

    def clear(self) -> None:
        self.history.clear()

    def answer(self, prompt: str, *, on_token: Callable[[str], None] | None = None
               ) -> GenerationResult:
        # Conservative UTF-8 byte budget, not a claimed tokenizer measurement.
        budget = self.context_tokens - self.max_tokens - 128
        if not prompt.strip() or len(prompt.encode("utf-8")) > budget:
            raise ValueError("prompt is empty or exceeds the conservative input budget")
        if self.client.model_digest(self.model) != self.digest:
            raise ValueError("model digest changed; restart the session explicitly")
        retained = list(self.history)
        self.truncated = False
        while retained and sum(len(t["content"].encode("utf-8")) + 16 for t in retained) + len(
            prompt.encode("utf-8")
        ) > budget:
            del retained[:2]
            self.truncated = True
        result = self.client.generate(model=self.model, prompt=prompt, history=retained,
                                      max_tokens=self.max_tokens,
                                      context_tokens=self.context_tokens, on_token=on_token)
        if self.client.model_digest(self.model) != self.digest:
            raise ValueError("model digest changed during generation; response discarded")
        if not result.content.strip():
            raise ValueError("backend returned an empty answer")
        self.history = retained + [{"role": "user", "content": prompt},
                                   {"role": "assistant", "content": result.content}]
        return result
