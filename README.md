# Hermes Local — experimental, release preparation

Hermes Local is a local-assistant integration being extracted from Olympus.
The current development implementation uses a separately installed Ollama base
model. It is **not** a trained Hermes 12B checkpoint, and this repository does not
currently contain an installable release or model weights.

## What has been demonstrated

In the developer's Olympus checkout, the terminal integration streamed real
responses using `qwen3:0.6b`, retained a word across two conversation turns, and
supported clearing and exiting the conversation. It includes explicit base-model
selection, model-digest checks and incomplete-stream handling.

These are integration smoke checks, not independent capability or safety evaluations.
The successful Qwen3 run does not establish the intelligence of a new Hermes model.
The parent checkout's test results are not certification of this new repository.

## What is not released

- Standalone installable source/package and clean-machine verification.
- A trained or fine-tuned Hermes checkpoint.
- Generative document-grounding with verified citations.
- A browser chat application or hosted service.
- Production, safety, general-intelligence or publication qualification.

## Release conditions

Before a usable release, the maintainer must approve code licensing, extract and
test the standalone implementation, document supported hardware and setup, and
publish versioned artifacts with accurate limitations. Model weights are acquired
separately under their publisher's license; none are redistributed here.

This repository is public for transparency during preparation. **Public visibility
is not a claim of launch readiness and does not grant a software license.** No
additional rights to the Olympus implementation are granted by this README.

The proposed initial product is an experimental assistant powered by a named
third-party base model, not an independently pretrained model. See the
[parent project](https://github.com/build-the-future-11/olympus-cognitive-architecture)
for the broader research context; development changes may not yet be published there.
