"""ComfyUI output node that uploads generated images to Immich with full metadata."""

import io
import json
import os
import time
import uuid
from datetime import datetime, timezone
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

        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), "wb") as f:
            f.write(png_bytes)

        return {"filename": filename, "subfolder": "", "type": "output"}

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
        """Add an asset to an Immich album."""
        body = json.dumps({"ids": [asset_id]}).encode()
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._api_request(f"{immich_url}/api/albums/{album_id}/assets", "PUT", headers, body)

    _POSITIVE_TITLES = {
        "@prompt",
        "@positive",
        "positive",
        "positive prompt",
    }
    _NEGATIVE_TITLES = {
        "@negative",
        "negative",
        "negative prompt",
        "negative (zeroed)",
    }
    _SKIP_PROMPT_TITLES = {"hand positive prompt"}

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
        immich_url, api_key = self._get_config()

        # Auto-build description from prompt data when none provided
        if not description:
            description = self._build_auto_description(prompt, character=character)

        results = []
        batch_size = images.shape[0]
        # One timeout per batch, not per image: if the first asset never leaves
        # upload/, the storage template engine is off and waiting is pointless.
        settle_enabled = True
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"[SaveToImmich] Uploading {batch_size} image(s) to {immich_url}")

        for i in range(batch_size):
            image_result = None
            try:
                png_bytes = self._build_png_bytes(
                    images[i], prompt=prompt, extra_pnginfo=extra_pnginfo
                )

                filename = f"{filename_prefix}_{timestamp}_{i:04d}_{uuid.uuid4().hex[:8]}.png"

                # Local preview first. Atelier and the ComfyUI UI both read
                # this file from history; Immich is the archive, not delivery.
                image_result = self._save_comfy_preview(png_bytes, filename) or {
                    "filename": filename
                }
                results.append(image_result)

                asset_id = self._upload_asset(immich_url, api_key, png_bytes, filename)
                if not asset_id:
                    print(
                        f"[SaveToImmich] WARNING: Upload {i + 1}/{batch_size} returned no asset ID"
                    )
                    continue

                print(f"[SaveToImmich] Uploaded {i + 1}/{batch_size}: {filename} -> {asset_id}")
                image_result["asset_id"] = asset_id

                if description:
                    try:
                        if settle_enabled and not self._wait_for_storage_settle(
                            immich_url, api_key, asset_id
                        ):
                            settle_enabled = False
                            print(
                                "[SaveToImmich] NOTE: asset did not leave upload/ within "
                                f"{_SETTLE_TIMEOUT_SECONDS:g}s; setting descriptions "
                                "immediately for the rest of this batch"
                            )
                        self._set_description(immich_url, api_key, asset_id, description)
                    except (HTTPError, URLError) as e:
                        print(
                            "[SaveToImmich] WARNING: Failed to set description: "
                            f"{_format_request_error(e)}"
                        )

                if album_id:
                    try:
                        self._add_to_album(immich_url, api_key, album_id, asset_id)
                    except (HTTPError, URLError) as e:
                        print(
                            "[SaveToImmich] WARNING: Failed to add to album: "
                            f"{_format_request_error(e)}"
                        )

            except (HTTPError, URLError) as e:
                print(
                    f"[SaveToImmich] ERROR: Failed to upload image {i + 1}/{batch_size}: "
                    f"{_format_request_error(e)}"
                )
            except Exception as e:
                print(f"[SaveToImmich] ERROR: Unexpected error on image {i + 1}/{batch_size}: {e}")
                if image_result is None:
                    continue

        uploaded = sum(1 for item in results if item.get("asset_id"))
        print(
            f"[SaveToImmich] Done. {len(results)}/{batch_size} preview(s), "
            f"{uploaded}/{batch_size} uploaded."
        )
        return {"ui": {"images": results}}
