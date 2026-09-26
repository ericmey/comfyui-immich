# Immich archive proof

This proof queues one frozen `EmptyImage` → `SaveToImmich` graph through a real
ComfyUI server. It compares the ComfyUI history receipt and downloaded preview
against Immich's independent asset readback and original PNG bytes. The input
image is a fixed 32 × 32 color square, so no checkpoint, GPU, or aesthetic
judgment is involved. One run proves that path only; it does not measure a
success rate or replace Eric's clean Manager install test.

`fixture.json` is the frozen input. The runner records its SHA-256, the exact
submitted graph SHA-256, prompt ID, asset ID, preview/original hashes, each
mechanical check, and failures. The unique run ID in the description prevents
ComfyUI from reusing a cached output. The installed node revision is an
**operator-declared** label; verify it separately on the ComfyUI host before
claiming a particular build was tested. The receipt omits endpoint URLs and
API keys. The asset remains in Immich for later inspection.

Run with the repository's Python environment (Pillow required). Supply an
Immich key with asset read permission via `IMMICH_READ_KEY`, keeping the key out
of shell history and committed files:

```bash
IMMICH_READ_KEY="..." .venv/bin/python evals/run_archive_proof.py \
  --comfy-url http://COMFY_HOST:8188 \
  --immich-url https://YOUR_IMMICH_HOST \
  --installed-node-ref COMMIT_SHA \
  --output /tmp/immich-archive-proof.json
```

The ComfyUI node itself needs its own configured upload key. The readback key
may have a different scope. The result is a pass only when every check in the
JSON receipt is true and no run error was recorded. Preserve failures too.

## Preliminary run

[`results/preflight-v0.3.0.json`](results/preflight-v0.3.0.json) is a successful
eight-check run against the private ComfyUI host's installed `a2cd0ec` build
(v0.3.0). The installed git head was read back separately from that host. The
ComfyUI history reported one uploaded asset and successful description, and
the downloaded preview matched Immich's original PNG byte-for-byte. The
embedded graph, dimensions, asset ID, and description matched the submission.
This establishes the proof path on **v0.3.0**; it does not test the changes
waiting in PRs #16 and #17.

## Failure and recovery

`fixtures/archive-proof.png` is a real ComfyUI preview with an embedded graph.
`run_recovery_proof.py` adds a unique description marker to its metadata, then
passes those **same PNG bytes** to `retry_archive.retry` twice. The first call
uses a deliberately refused local endpoint and must return an `upload` failure.
The second uses the configured Immich service and must return an asset ID. The
runner reads that asset back and compares its original bytes and description
with the trial PNG. It records the base fixture hash, trial hash, run ID, each
stage receipt, and all checks; no GPU render happens between attempts.

```bash
IMMICH_WRITE_KEY="..." IMMICH_READ_KEY="..." \
  .venv/bin/python evals/run_recovery_proof.py \
  --immich-url https://YOUR_IMMICH_HOST \
  --source-ref COMMIT_SHA \
  --output /tmp/immich-recovery-proof.json
```

[`results/preflight-recovery-v0.4.2.json`](results/preflight-recovery-v0.4.2.json)
passes six checks against the real Immich service, using the checkout at
`b22de49` (v0.4.2). The recovered asset ID differed from the earlier archive
proof asset. This exercises the local retry module, **not** a deployed ComfyUI
node, and the initial failure is induced. It does not establish reliability
under a real outage or the changes waiting in PRs #16/#17.

**Boundary:** the runner executes the deployed ComfyUI node. A source checkout
or green unit test does not prove those bytes are deployed. A separate
post-PR archive run and the new Settings panel still need their own receipts.
