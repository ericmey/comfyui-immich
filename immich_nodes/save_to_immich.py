"""ComfyUI output node that uploads generated images to Immich with full metadata."""

import io
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import ClassVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

try:
    import folder_paths
except ImportError:
    folder_paths = None

# Immich stores a freshly uploaded asset under upload/ and the storage template
# engine then moves it into library/. The sidecar write queued by a description
# PUT reads the asset path when its handler starts, so setting a description
# before that move leaves the handler racing it to a path that is about to go.
_UPLOAD_PATH_MARKER = "/upload/"
_SETTLE_TIMEOUT_SECONDS = 5.0
_SETTLE_POLL_SECONDS = 0.25

# Stage failures worth reporting as a receipt rather than crashing the graph:
# transport, filesystem and bad-response errors. URLError, HTTPError and
# TimeoutError are all OSError subclasses; JSONDecodeError is a ValueError.
# Anything outside this tuple is a bug in this node and must keep propagating.
_ARCHIVE_ERRORS = (OSError, ValueError)


def _load_env(env_path):
    """Parse a .env file into a dict. Skips comments and blank lines."""
    env = {}
    if not os.path.isfile(env_path):
        return env
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip().strip("\"'")
    return env


def _normalize_immich_url(url):
    """Normalize Immich base URL values from .env or shell environment."""
    url = (url or "").strip().rstrip("/")
    if url.endswith("/api"):
        url = url[: -len("/api")]
    return url


def _multipart_encode(fields, files):
    """Build a multipart/form-data body.

    fields: list of (name, value) tuples
    files:  list of (name, filename, content_type, data_bytes) tuples

    Returns (body_bytes, content_type).
    """
    boundary = uuid.uuid4().hex
    lines = []

    for name, value in fields:
        lines.append(f"--{boundary}".encode())
        lines.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        lines.append(b"")
        lines.append(value.encode() if isinstance(value, str) else value)

    for name, filename, content_type, data in files:
        if any(c in filename for c in '\r\n\x00"'):
            raise ValueError("invalid upload filename")
        lines.append(f"--{boundary}".encode())
        lines.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"'.encode()
        )
        lines.append(f"Content-Type: {content_type}".encode())
        lines.append(b"")
        lines.append(data)

    lines.append(f"--{boundary}--".encode())
    lines.append(b"")

    body = b"\r\n".join(lines)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def _new_archive_report(filename, description, album_id):
    """Build the per-image receipt every archive path fills in."""
    return {
        "filename": filename,
        "upload": "failed",
        "description": "not_requested" if not description else "not_attempted",
        "album": "not_requested" if not album_id else "not_attempted",
        "errors": [],
    }


def _print_report_errors(report):
    for error in report["errors"]:
        print(f"[SaveToImmich] {error['stage']} failed: {error['message']}")


def _format_request_error(error):
    """Return a useful HTTP error string without logging request headers."""
    if isinstance(error, HTTPError):
        try:
            body = error.read().decode("utf-8", errors="replace").strip()
        except Exception:
            body = ""

        status = f"HTTP {error.code}"
        if error.reason:
            status = f"{status} {error.reason}"
        return f"{status}: {body}" if body else status

    return str(error)


class SaveToImmich:
    """ComfyUI output node that uploads images to Immich with embedded metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "character": ("STRING", {"default": ""}),
                "description": ("STRING", {"default": "", "multiline": True}),
                "album_id": ("STRING", {"default": ""}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "upload"
    CATEGORY = "image/immich"

    def _get_config(self):
        """Load IMMICH_URL and IMMICH_API_KEY from .env in the package root."""
        package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env_path = os.path.join(package_dir, ".env")
        env = _load_env(env_path)

        immich_url = os.environ.get("IMMICH_URL") or env.get("IMMICH_URL", "")
        api_key = os.environ.get("IMMICH_API_KEY") or env.get("IMMICH_API_KEY", "")

        if not immich_url:
            raise ValueError(
                f"IMMICH_URL not set. Create a .env file at {env_path} "
                "with IMMICH_URL=https://your-immich-instance.com"
            )
        if not api_key:
            raise ValueError(
                f"IMMICH_API_KEY not set. Create a .env file at {env_path} "
                "with IMMICH_API_KEY=your-api-key-here"
            )

        return _normalize_immich_url(immich_url), api_key.strip()

    def _api_request(self, url, method, headers, body=None):
        """Make an HTTP request and return parsed JSON response."""
        req = Request(url, data=body, headers=headers, method=method)
        with urlopen(req, timeout=30) as resp:
            data = resp.read()
            if not data:
                return {}
            return json.loads(data.decode())

    def _build_png_bytes(self, img_tensor, prompt=None, extra_pnginfo=None):
        """Convert image tensor to PNG bytes with embedded metadata.

        Embeds the full ComfyUI workflow and prompt data into the PNG,
        matching the behavior of ComfyUI's built-in SaveImage node.
        """
        img_array = (img_tensor.cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_array)

        metadata = PngInfo()

        # Embed prompt (node inputs) — same key ComfyUI uses
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))

        # Embed workflow and any extra PNG info
        if extra_pnginfo is not None:
            for key, value in extra_pnginfo.items():
                metadata.add_text(key, json.dumps(value))

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG", pnginfo=metadata)
        return buf.getvalue()

    def _upload_asset(self, immich_url, api_key, png_bytes, filename):
        """Upload a single image to Immich. Returns the asset ID."""
        now = datetime.now(timezone.utc).isoformat()

        fields = [
            ("fileCreatedAt", now),
            ("fileModifiedAt", now),
        ]
        files = [
            ("assetData", filename, "image/png", png_bytes),
        ]

        body, content_type = _multipart_encode(fields, files)
        headers = {
            "x-api-key": api_key,
            "Content-Type": content_type,
            "Accept": "application/json",
        }

        result = self._api_request(f"{immich_url}/api/assets", "POST", headers, body)
        return result.get("id")

    def _save_comfy_preview(self, png_bytes, filename):
        """Save a ComfyUI-viewable local preview and return frontend image metadata."""
        if folder_paths is None:
            return None

        relative = Path(filename.replace("\\", "/"))
        if (
            relative.is_absolute()
            or PureWindowsPath(filename).drive
            or ".." in relative.parts
            or any(c in filename for c in '\r\n\x00"')
        ):
            raise ValueError("filename_prefix must stay within the ComfyUI output directory")
        output_dir = Path(folder_paths.get_output_directory()).resolve()
        target = (output_dir / relative).resolve()
        if output_dir not in target.parents:
            raise ValueError("filename_prefix escapes the ComfyUI output directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write(png_bytes)
        subfolder = (
            "" if target.parent == output_dir else target.parent.relative_to(output_dir).as_posix()
        )
        return {"filename": target.name, "subfolder": subfolder, "type": "output"}

    def _get_asset(self, immich_url, api_key, asset_id):
        """Fetch a single asset. Used to observe where Immich has put the file."""
        headers = {
            "x-api-key": api_key,
            "Accept": "application/json",
        }
        return self._api_request(f"{immich_url}/api/assets/{asset_id}", "GET", headers)

    def _wait_for_storage_settle(self, immich_url, api_key, asset_id):
        """Block until Immich has moved the asset out of upload/.

        The description PUT queues a sidecar write, and that job reads the
        asset's originalPath when the handler starts. Writing the description
        while the file is still in upload/ races the storage template move: the
        handler reads the pre-move path, the move lands underneath it, and the
        stat fails with ENOENT. The asset itself is fine; the .xmp sidecar is
        what gets lost.

        StorageCore.moveFile renames the file before saving the new path, so an
        originalPath outside upload/ proves the physical move already finished.

        Returns True once the asset has moved, False if it never did within the
        timeout — which is also what happens when the storage template engine is
        switched off and the file legitimately stays in upload/ forever.
        """
        deadline = time.monotonic() + _SETTLE_TIMEOUT_SECONDS
        while True:
            try:
                asset = self._get_asset(immich_url, api_key, asset_id)
            except (HTTPError, URLError):
                # A transient lookup failure says nothing about where the file
                # is. Treating it as settled would fail open into the very race
                # this wait exists to close, so keep trying until the deadline.
                asset = None
            if asset is not None:
                path = str(asset.get("originalPath") or "")
                if path and _UPLOAD_PATH_MARKER not in path:
                    return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(_SETTLE_POLL_SECONDS)

    def _set_description(self, immich_url, api_key, asset_id, description):
        """Set the description on an Immich asset."""
        body = json.dumps({"description": description}).encode()
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._api_request(f"{immich_url}/api/assets/{asset_id}", "PUT", headers, body)

    def _add_to_album(self, immich_url, api_key, album_id, asset_id):
        """Add an asset to an Immich album. Returns True if Immich confirmed it."""
        body = json.dumps({"ids": [asset_id]}).encode()
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        result = self._api_request(
            f"{immich_url}/api/albums/{album_id}/assets", "PUT", headers, body
        )
        # Immich reports per-asset failures in a successful HTTP response, so a
        # 2xx alone proves nothing. But a response with no per-asset body at all
        # (a 204, or a proxy that strips it) carries no failure either: report
        # that as unconfirmed rather than inventing either outcome.
        if not isinstance(result, list):
            return False
        entries = [item for item in result if isinstance(item, dict) and item.get("id") == asset_id]
        if len(entries) != 1 or not (
            entries[0].get("success") is True or entries[0].get("error") == "duplicate"
        ):
            raise ValueError("Immich did not confirm album membership for this asset")
        return True

    _POSITIVE_TITLES: ClassVar[set[str]] = {
        "@prompt",
        "@positive",
        "positive",
        "positive prompt",
    }
    _NEGATIVE_TITLES: ClassVar[set[str]] = {
        "@negative",
        "negative",
        "negative prompt",
        "negative (zeroed)",
    }
    _SKIP_PROMPT_TITLES: ClassVar[set[str]] = {"hand positive prompt"}

    def _build_auto_description(self, prompt, character=""):
        """Build a description from the ComfyUI workflow prompt data.

        Titles match both Atelier's normaliser ("Positive prompt",
        "Diffusion model") and common hand-built graphs (@positive,
        CheckpointLoaderSimple).
        """
        if not prompt:
            return f"Character: {character}" if character else ""

        lines = []
        if character:
            lines.append(f"Character: {character}")
        nodes = prompt if isinstance(prompt, dict) else {}

        positive_text = ""
        negative_text = ""
        checkpoint = ""
        seed = ""
        sampler = ""
        steps = ""
        cfg = ""

        for node in nodes.values():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {})
            title = str(node.get("_meta", {}).get("title", "")).strip().lower()

            if class_type == "CLIPTextEncode":
                text = inputs.get("text", "")
                if not text:
                    continue
                if title in self._NEGATIVE_TITLES:
                    negative_text = negative_text or text
                elif title in self._POSITIVE_TITLES:
                    positive_text = positive_text or text
                elif not positive_text and title not in self._SKIP_PROMPT_TITLES:
                    positive_text = text

            elif class_type in ("CheckpointLoaderSimple", "UNETLoader"):
                checkpoint = inputs.get("ckpt_name") or inputs.get("unet_name") or checkpoint

            elif class_type == "KSampler" or (
                isinstance(class_type, str) and "KSampler" in class_type
            ):
                seed = str(inputs.get("seed", seed))
                sampler = inputs.get("sampler_name", sampler) or sampler
                steps = str(inputs.get("steps", steps))
                cfg = str(inputs.get("cfg", cfg))

        if checkpoint:
            lines.append(f"Checkpoint: {checkpoint}")
        if sampler:
            lines.append(f"Sampler: {sampler} | Steps: {steps} | CFG: {cfg}")
        if seed:
            lines.append(f"Seed: {seed}")
        if positive_text:
            lines.append(f"\nPositive: {positive_text}")
        if negative_text:
            lines.append(f"\nNegative: {negative_text}")

        return "\n".join(lines)

    def archive_png(
        self,
        png_bytes,
        filename,
        *,
        description="",
        album_id="",
        asset_id=None,
        wait_for_settle=True,
    ):
        """Archive existing PNG bytes and report each stage without losing delivery."""
        report = _new_archive_report(filename, description, album_id)
        reused = bool(asset_id)
        try:
            immich_url, api_key = self._get_config()
            if not reused:
                asset_id = self._upload_asset(immich_url, api_key, png_bytes, filename)
            if not asset_id:
                raise ValueError("Immich returned no asset ID")
        except _ARCHIVE_ERRORS as exc:
            report["errors"].append({"stage": "upload", "message": _format_request_error(exc)})
            return report
        # "reused" and "ok" are not the same claim: nothing was uploaded here.
        report.update(upload="reused" if reused else "ok", asset_id=asset_id)
        if description:
            try:
                if wait_for_settle:
                    report["storage_settled"] = self._wait_for_storage_settle(
                        immich_url, api_key, asset_id
                    )
                self._set_description(immich_url, api_key, asset_id, description)
                report["description"] = "ok"
            except _ARCHIVE_ERRORS as exc:
                report["description"] = "failed"
                report["errors"].append(
                    {"stage": "description", "message": _format_request_error(exc)}
                )
        if album_id:
            try:
                confirmed = self._add_to_album(immich_url, api_key, album_id, asset_id)
                report["album"] = "ok" if confirmed else "unconfirmed"
            except _ARCHIVE_ERRORS as exc:
                report["album"] = "failed"
                report["errors"].append({"stage": "album", "message": _format_request_error(exc)})
        return report

    def _archive_image(
        self,
        tensor,
        filename,
        *,
        prompt,
        extra_pnginfo,
        description,
        album_id,
        wait_for_settle,
    ):
        """Encode, deliver and archive one image. Returns (report, preview)."""
        basename = Path(filename).name
        try:
            png_bytes = self._build_png_bytes(tensor, prompt=prompt, extra_pnginfo=extra_pnginfo)
        except _ARCHIVE_ERRORS as exc:
            report = _new_archive_report(basename, description, album_id)
            report["errors"].append({"stage": "encode", "message": _format_request_error(exc)})
            return report, None

        # A rejected filename_prefix or an unwritable output directory is a
        # delivery failure, not an archive failure. Immich still gets the image.
        preview = None
        preview_error = None
        try:
            preview = self._save_comfy_preview(png_bytes, filename)
        except _ARCHIVE_ERRORS as exc:
            preview_error = {"stage": "preview", "message": _format_request_error(exc)}

        report = self.archive_png(
            png_bytes,
            basename,
            description=description,
            album_id=album_id,
            wait_for_settle=wait_for_settle,
        )
        if preview_error:
            report["errors"].insert(0, preview_error)
        if preview is not None:
            report.update(filename=preview["filename"], subfolder=preview.get("subfolder", ""))
            if report.get("asset_id"):
                preview["asset_id"] = report["asset_id"]
        return report, preview

    def upload(
        self,
        images,
        character="",
        description="",
        album_id="",
        filename_prefix="ComfyUI",
        prompt=None,
        extra_pnginfo=None,
    ):
        if not description:
            description = self._build_auto_description(prompt, character=character)
        results = []
        reports = []
        settle_enabled = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i in range(images.shape[0]):
            filename = f"{filename_prefix}_{timestamp}_{i:04d}_{uuid.uuid4().hex[:8]}.png"
            # Last-resort containment. Images earlier in the batch may already
            # be in Immich, so an error escaping here would strand exactly the
            # asset IDs a retry needs. Even a bug becomes a receipt.
            try:
                report, preview = self._archive_image(
                    images[i],
                    filename,
                    prompt=prompt,
                    extra_pnginfo=extra_pnginfo,
                    description=description,
                    album_id=album_id,
                    wait_for_settle=settle_enabled,
                )
            except Exception as exc:
                report = _new_archive_report(Path(filename).name, description, album_id)
                report["errors"].append(
                    {"stage": "unexpected", "message": f"{type(exc).__name__}: {exc}"}
                )
                preview = None

            if report.get("storage_settled") is False:
                settle_enabled = False
            if preview is not None:
                results.append(preview)
            reports.append(report)
            _print_report_errors(report)
        return {"ui": {"images": results, "archive": reports}}
