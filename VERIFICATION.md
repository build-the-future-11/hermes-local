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
