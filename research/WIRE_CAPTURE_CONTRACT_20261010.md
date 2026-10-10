# Bounded provider wire capture — prospective engineering contract

Written before implementation or new regression outcomes on 10 October 2026.
Parent source: `e5475f01ac3aa2c9614c9440a89f9026749ea152` (draft PR #2).

## Purpose and observed gap

The parent checks final answer length after `response.json()` or `iter_lines()`
has already buffered provider data. Large unused JSON fields and an unbroken
stream line can exceed the intended client budget before that check. Default
JSON decoding also collapses duplicate model/completion fields and accepts
nonfinite numeric constants. These are provider transport and identity defects,
not evidence about the quality of Hermes or any third-party model.

## Intended implementation

Both ordinary HTTP responses and streamed chat must enter through a byte-budgeted
reader. Limit an entire response to 16 MiB, a streamed JSON record to 8 MiB, and
the stream to 16,384 records, counting blank lines. Keep the existing 1 MiB UTF-8
answer limit and all parent PR #2 completion, model and usage checks. Apply bounds
before accumulating each incoming segment or parsing an object. Close response
handles on bounds, malformed bytes, callback failure and normal completion.

Request `Accept-Encoding: identity` and refuse nonidentity content encoding before
body iteration; do not permit transparent decompression to expand bytes ahead of
the client budget. Normal JSON and newline-delimited JSON are UTF-8. Accept LF,
CRLF, arbitrary transport chunk boundaries and a final record without newline.
Reject duplicate object keys at any depth, nonfinite constants/overflow, invalid
UTF-8, and nonobject records. A malformed record must not reach a token callback.

Only small bounded byte buffers are application-owned. The HTTP transport can
already allocate a chunk before handing it to this client; this contract does
not claim control of allocations inside third-party transports or the server.
The existing HTTP per-operation timeout remains; there is no new total wall-clock
deadline or promise of immediate server cancellation. Partial displayed text is
still provisional until a valid completion receipt is obtained. Failed requests
preserve existing chat history and never write durable conversation data.

## Verification and boundary

Use instrumented `httpx.SyncByteStream` and `MockTransport` only. Check exact
boundary acceptance/refusal, a never-terminated record, oversize ignored metadata,
empty-record floods, split UTF-8/CRLF, duplicate identities, finite JSON, response
closure, failed history preservation and unchanged parent completion behavior.
The full existing suite remains required. No local model, live network endpoint,
external integration, research outcome, training campaign or paid job is run.
Model qualification, generative citation validation and the existing proprietary
license remain unchanged. Scientific completion is not established by these tests.
