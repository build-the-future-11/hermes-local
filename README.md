# Hermes Local — experimental developer preview

A runnable terminal assistant integration powered by a separately installed Ollama
model. **Not an independently trained Hermes 12B model, not production-qualified,
and not a browser application.** No model weights are bundled.

## License first

This source retains the Olympus Proprietary License. Public visibility does not
grant permission to copy, modify or redistribute it. Installation/use instructions
below are for the copyright holder or users with separate written permission.
This is not an open-source or generally licensed public-use release. The maintainer
must authorize broader usage before promoting it as one.

## Installation (authorized users)

Requires Python 3.14+, pip, and a separately installed Ollama runtime.
Chat does not require PyTorch. Tested on macOS; other environments are unverified.

```sh
git clone https://github.com/build-the-future-11/hermes-local.git
cd hermes-local
python3.14 -m venv .venv
.venv/bin/python -m pip install .
ollama serve
```

In another terminal, acquire a model under its publisher's terms, then run:

```sh
ollama pull qwen3:0.6b
.venv/bin/hermes doctor --model qwen3:0.6b
.venv/bin/hermes chat --model qwen3:0.6b
```

The example downloads roughly 523 MB. See the
[official model listing](https://ollama.com/library/qwen3:0.6b) for details and license.
The base model is third-party software, not Hermes weights. No paid API is required.
Choose a model that fits your available memory; no universal hardware guarantee is made.

Use `--base-url http://127.0.0.1:11435` for a separately configured local instance.
An Ollama API address is not a chat website.

## Behavior and limits

- Streamed multi-turn terminal chat; `/clear`, `/exit`, Ctrl-C/EOF.
- Explicit model selection and before/after digest checks; no silent fallback.
- Bounded history with visible omission notices; conservative byte budgeting,
  not an exact tokenizer calculation.
- Streaming and non-streaming answers require an explicit `done: true` receipt
  with `done_reason: "stop"` or `"length"`. Missing or unsupported reasons fail
  explicitly; a token limit remains visible as `length`.
- Failed/incomplete responses are not saved to history. Final stream metadata is
  checked before that event's text is displayed. Earlier displayed partial text
  must be discarded on failure. Client disconnect does not certify immediate
  server cancellation.
- Response text is bounded to 1 MiB of UTF-8 bytes in both modes. Installed model
  names must be unambiguous before their digests can identify a chat session.
- No application conversation persistence or tool execution. Backend logging is separate.
- Remote backend selection sends conversation contents there; HTTPS is required.
- Chat answers are not document-verified and may be incorrect.

## Full Hermes research source

`hermes_local/grounding.py` includes the extractive grounding service, trainable
heads and workspace runtime. `substrate.py` and `core.py` supply their original
typed support contracts. These are experimental components, not trained weights.

```sh
.venv/bin/python -m pip install '.[research,test]'
.venv/bin/python -m pytest
```

The grounding service is separate from generative chat. Generative citation
validation, model qualification, independent evaluation, durable installation
management and a browser UI are not included.

## Provenance

Extracted from the owner's Olympus development checkout. Original modules were
preserved with package imports remapped; a standalone CLI and packaging were added.
Parent-project test counts do not certify this repository. See `VERIFICATION.md`
for checks actually run on this extraction.
