"""Run one real ComfyUI -> Immich archive and mechanically check both copies.

No model or GPU is needed: EmptyImage provides a fixed 32-pixel square. This
proves the output node's archive path, not image generation or installation.
"""

import argparse
import hashlib
import io
import json
import os
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

FIXTURE_PATH = Path(__file__).with_name("fixture.json")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def scrub(value, private_values):
    """Keep endpoint addresses and the read key out of shareable receipts."""
    if isinstance(value, str):
        for index, private in enumerate(private_values):
            if private:
                value = value.replace(private, f"<private-{index}>")
        return value
    if isinstance(value, dict):
        return {key: scrub(item, private_values) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item, private_values) for item in value]
    return value


def request_json(url, *, key=None, data=None):
    headers = {"Accept": "application/json"}
    if key:
        headers["x-api-key"] = key
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, headers=headers, data=data)
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.load(response)


def request_bytes(url, *, key=None):
    headers = {"x-api-key": key} if key else {}
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=20
    ) as response:
        return response.read()


def wait_history(base, prompt_id, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = request_json(f"{base}/history/{prompt_id}")
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(0.5)
    raise TimeoutError(f"ComfyUI history did not contain prompt {prompt_id} within {timeout}s")


def wait_description(base, key, asset_id, expected, timeout=20):
    deadline = time.monotonic() + timeout
    asset = {}
    while time.monotonic() < deadline:
        asset = request_json(f"{base}/api/assets/{asset_id}", key=key)
        if (asset.get("exifInfo") or {}).get("description") == expected:
            return asset
        time.sleep(0.5)
    return asset


def run(args):
    fixture_bytes = FIXTURE_PATH.read_bytes()
    fixture = json.loads(fixture_bytes)
    comfy = args.comfy_url.rstrip("/")
    immich = args.immich_url.rstrip("/")
    key = os.environ.get("IMMICH_READ_KEY", "")
    if not key:
        raise ValueError("IMMICH_READ_KEY must be set for independent Immich readback")

    run_id = uuid.uuid4().hex[:12]
    description = f"{fixture['description_prefix']} {run_id}"
    graph = {
        "1": {
            "class_type": "EmptyImage",
            "inputs": {name: fixture[name] for name in ("width", "height", "batch_size", "color")},
        },
        "2": {
            "class_type": "SaveToImmich",
            "inputs": {
                "images": ["1", 0],
                "description": description,
                "filename_prefix": fixture["filename_prefix"],
            },
        },
    }
    receipt = {
        "schema": "immich-archive-proof-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "installed_node_ref_operator_declared": args.installed_node_ref,
        "fixture_sha256": sha256(fixture_bytes),
        "runner_sha256": sha256(Path(__file__).read_bytes()),
        "submitted_graph_sha256": sha256(json.dumps(graph, sort_keys=True).encode()),
        "run_id": run_id,
        "checks": {},
        "limits": [
            "The installed node ref is operator-declared; server code bytes were not attested.",
            "One synthetic image is a path proof, not a reliability rate or a clean-install test.",
        ],
    }
    try:
        queued = request_json(
            f"{comfy}/prompt",
            data=json.dumps({"prompt": graph, "client_id": f"immich-eval-{run_id}"}).encode(),
        )
        prompt_id = queued["prompt_id"]
        receipt["prompt_id"] = prompt_id
        receipt["checks"]["queued"] = True
        history = wait_history(comfy, prompt_id, args.timeout)
        receipt["history_status"] = history.get("status", {}).get("status_str")
        output = history.get("outputs", {}).get("2", {})
        archive = output.get("archive", [])
        images = output.get("images", [])
        report = archive[0] if len(archive) == 1 else {}
        preview = images[0] if len(images) == 1 else {}
        receipt["archive_report"] = report
        receipt["preview_report"] = preview
        receipt["checks"]["history_archive_receipt"] = (
            len(archive) == 1 and report.get("upload") == "ok" and report.get("description") == "ok"
        )
        receipt["checks"]["history_preview_receipt"] = len(images) == 1 and bool(
            preview.get("filename")
        )
        asset_id = report.get("asset_id")
        if not (asset_id and preview.get("filename")):
            raise ValueError("history has no asset ID or preview filename")

        query = urllib.parse.urlencode(
            {
                "filename": preview["filename"],
                "subfolder": preview.get("subfolder", ""),
                "type": "output",
            }
        )
        preview_bytes = request_bytes(f"{comfy}/view?{query}")
        original_bytes = request_bytes(f"{immich}/api/assets/{asset_id}/original", key=key)
        receipt["preview_sha256"] = sha256(preview_bytes)
        receipt["immich_original_sha256"] = sha256(original_bytes)
        receipt["checks"]["asset_bytes_equal_preview"] = preview_bytes == original_bytes
        with Image.open(io.BytesIO(original_bytes)) as image:
            receipt["checks"]["image_dimensions"] = image.size == (
                fixture["width"],
                fixture["height"],
            )
            embedded_prompt = json.loads(image.info.get("prompt", "null"))
        receipt["checks"]["embedded_graph_equal_submitted"] = embedded_prompt == graph
        asset = wait_description(immich, key, asset_id, description)
        receipt["asset_id"] = asset_id
        receipt["checks"]["immich_asset_id"] = asset.get("id") == asset_id
        receipt["checks"]["immich_description"] = (asset.get("exifInfo") or {}).get(
            "description"
        ) == description
        receipt["asset_original_filename"] = asset.get("originalFileName")
    except Exception as exc:
        receipt["run_error"] = f"{type(exc).__name__}: {exc}"
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["passed"] = (
        bool(receipt["checks"]) and all(receipt["checks"].values()) and not receipt.get("run_error")
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_receipt = scrub(receipt, (key, comfy, immich))
    output_path.write_text(json.dumps(safe_receipt, indent=2, sort_keys=True) + "\n")
    print(f"{'PASS' if receipt['passed'] else 'FAIL'}: {output_path}")
    return 0 if receipt["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-url", required=True)
    parser.add_argument("--immich-url", required=True)
    parser.add_argument("--installed-node-ref", required=True)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--output", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
