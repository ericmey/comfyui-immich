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

**Boundary:** the runner executes the deployed ComfyUI node. A source checkout
or green unit test does not prove those bytes are deployed. A separate
failure/retry proof and the new Settings panel still need their own receipts.
