# Verification of the standalone extraction

Local macOS / Python 3.14 verification:

- 10 tests passed: chat history, bounds, digest changes, incomplete streaming,
  interruption, extractive citations/authorization and trainable grounding components.
- Ruff passed using the parent project's E/F/I/B/UP rule set.
- Wheel and source archive build passed.
- Wheel installed with no dependency installation into a fresh temporary target;
  an isolated interpreter outside the checkout loaded CLI help without importing
  Torch. Dependencies came from the existing interpreter; this is not a fresh-machine
  dependency-install certification.

Earlier real Qwen3 streaming and CLI results belong to the parent implementation.
No new standalone capability benchmark, clean-machine model download, remote CI pass,
production deployment or Hermes checkpoint qualification is claimed here.

Current limitations include unpinned dependency resolution, provisional streamed
text, server cancellation uncertainty, conservative context budgeting and missing
generative document citation validation. The proprietary license restricts use.

## 10 October 2026 — bounded wire capture

The separate wire-capture revision is based on completion-contract PR #2 at
`e5475f01ac3aa2c9614c9440a89f9026749ea152`. Its original records above remain valid
for their stated source scopes. The complete local Linux/Python 3.12.14 suite
passes **96 cases: 58 inherited and 38 new**, using HTTPX 0.28.1 and Pydantic 2.14.0.
All new provider checks use instrumented offline `MockTransport`/`SyncByteStream`.
Configured Ruff checks and patch whitespace checks pass.

Independent review repeated all 96 cases and tested four HTTP error/ReadError
closure paths. It inspected HTTPX's identity iteration to verify bounds precede
application accumulation and transparent decompression. Its initial default-Python
collection lacked HTTPX; the supplied isolated dependency path resolved this
environment issue. No project assertion was weakened. No new model generation,
training campaign, live provider endpoint or scientific result was executed.

The retained [machine-readable receipt](research/WIRE_CAPTURE_VERIFICATION_20261010.json)
includes source hashes, exact commands, full outputs, parent counterexamples and
passing controls. Hosted checks on the new published source are tracked in its
draft PR; local Python 3.12 execution alone does not certify the declared Python
3.14 packaging path.

## Provider completion contract — 2026-10-10

The original non-streaming client accepted `done: false` as a completed
length-limited answer and inferred successful termination from missing or unknown
reasons. Streaming validated final usage fields after displaying the event's text.
Duplicate installed model names were rejected by discovery but accepted by digest
lookup. The corrected paths now require explicit supported completion, validate
terminal metadata before callbacks, preserve session history on failure, and use
the same model-list admission for discovery and digest lookup.

Local Linux / Python 3.12.14 checks (a compatibility probe below the declared
Python 3.14 installation minimum):

- **58 tests passed** across the complete source suite, including 48 new provider
  receipt, streaming, boundary, identity, and session regressions.
- Replaying the new tests against the original backend at source commit
  `acc24d1134cb2d3aac5c66792321b75919b81fdc` produced **35 failures and 13 passes**.
- Supported `stop` and `length` receipts retain their classifications. UTF-8 size
  limits include a valid boundary case. Invalid final events emit no final text;
  previously emitted text remains provisional and failed turns never enter history.
- All provider interactions in these tests use `httpx.MockTransport`; no installed
  model or live generation quality is certified. The separate grounding tests
  used PyTorch 2.14.1+cpu. Changed Python files passed Ruff.

The repository's standalone GitHub workflow remains the Python 3.14 installation,
test, package-build, and CLI gate. Its result must be read for the actual revision;
the local results above do not imply a hosted pass or a newly qualified checkpoint.
