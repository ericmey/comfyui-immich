"""Force one archive failure, then retry the same ComfyUI PNG without rendering."""

import argparse
import hashlib
import io
import json
import os
import tempfile
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from immich_nodes.retry_archive import retry

FIXTURE = Path(__file__).parent / "fixtures" / "archive-proof.png"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def scrub(value, private_values):
    if isinstance(value, str):
        for private in private_values:
            if private:
                value = value.replace(private, "<private>")
        return value
    if isinstance(value, dict):
        return {key: scrub(item, private_values) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item, private_values) for item in value]
    return value


def readback(url, key, asset_id):
    def get(suffix):
        request = urllib.request.Request(
            f"{url.rstrip('/')}/api/assets/{asset_id}{suffix}",
            headers={"x-api-key": key},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.read()

    return json.loads(get("")), get("/original")


def run(args):
    write_key = os.environ.get("IMMICH_WRITE_KEY", "")
    read_key = os.environ.get("IMMICH_READ_KEY", "")
    if not write_key or not read_key:
        raise ValueError("IMMICH_WRITE_KEY and IMMICH_READ_KEY are required")
    base_png = FIXTURE.read_bytes()
    run_id = uuid.uuid4().hex[:12]
    with Image.open(FIXTURE) as image:
        graph = json.loads(image.info["prompt"])
        graph["2"]["inputs"]["description"] += f" recovery {run_id}"
        description = graph["2"]["inputs"]["description"]
        metadata = PngInfo()
        for name, value in image.info.items():
            if isinstance(value, str):
                metadata.add_text(name, json.dumps(graph) if name == "prompt" else value)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", pnginfo=metadata)
    png = buffer.getvalue()
    receipt = {
        "schema": "immich-recovery-proof-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_ref_operator_declared": args.source_ref,
        "base_fixture_sha256": sha256(base_png),
        "trial_png_sha256": sha256(png),
        "run_id": run_id,
        "runner_sha256": sha256(Path(__file__).read_bytes()),
        "checks": {},
        "limits": [
            "The first failure is induced by a refused local endpoint, not an observed outage.",
            "This runs the checkout's retry module directly, not a deployed ComfyUI node.",
        ],
    }
    old_url = os.environ.get("IMMICH_URL")
    old_key = os.environ.get("IMMICH_API_KEY")
    trial_dir = tempfile.TemporaryDirectory(prefix="immich-recovery-")
    trial_path = Path(trial_dir.name) / "archive-proof-retry.png"
    trial_path.write_bytes(png)
    try:
        os.environ["IMMICH_URL"] = "http://127.0.0.1:9"
        os.environ["IMMICH_API_KEY"] = "deliberately-invalid"
        failed = retry(trial_path)
        receipt["failed_attempt"] = {key: value for key, value in failed.items() if key != "errors"}
        receipt["checks"]["induced_upload_failure"] = failed["upload"] == "failed" and [
            error["stage"] for error in failed["errors"]
        ] == ["upload"]
        os.environ["IMMICH_URL"] = args.immich_url
        os.environ["IMMICH_API_KEY"] = write_key
        recovered = retry(trial_path)
        receipt["retry_report"] = recovered
        receipt["checks"]["retry_uploaded"] = recovered.get("upload") == "ok"
        receipt["checks"]["retry_description"] = recovered.get("description") == "ok"
        asset_id = recovered.get("asset_id")
        if not asset_id:
            raise ValueError("retry returned no asset ID")
        asset, original = readback(args.immich_url, read_key, asset_id)
        receipt["asset_id"] = asset_id
        receipt["original_sha256"] = sha256(original)
        receipt["checks"]["original_equals_trial_png"] = original == png
        receipt["checks"]["asset_id_readback"] = asset.get("id") == asset_id
        receipt["checks"]["description_readback"] = (asset.get("exifInfo") or {}).get(
            "description"
        ) == description
    except Exception as exc:
        message = str(exc)
        for secret in (write_key, read_key, args.immich_url):
            if secret:
                message = message.replace(secret, "<private>")
        receipt["run_error"] = f"{type(exc).__name__}: {message}"
    finally:
        trial_dir.cleanup()
        for name, value in (("IMMICH_URL", old_url), ("IMMICH_API_KEY", old_key)):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["passed"] = (
        bool(receipt["checks"]) and all(receipt["checks"].values()) and not receipt.get("run_error")
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    safe_receipt = scrub(receipt, (write_key, read_key, args.immich_url))
    output.write_text(json.dumps(safe_receipt, indent=2, sort_keys=True) + "\n")
    print(f"{'PASS' if receipt['passed'] else 'FAIL'}: {output}")
    return 0 if receipt["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--immich-url", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--output", required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
