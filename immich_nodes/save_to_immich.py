"""ComfyUI output node that uploads generated images to Immich with full metadata."""

import io
import json
import os
import uuid
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo


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


class SaveToImmich:
    """ComfyUI output node that uploads images to Immich with embedded metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
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

        immich_url = env.get("IMMICH_URL", "")
        api_key = env.get("IMMICH_API_KEY", "")

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

        return immich_url.rstrip("/"), api_key

    def _api_request(self, url, method, headers, body=None):
        """Make an HTTP request and return parsed JSON response."""
        req = Request(url, data=body, headers=headers, method=method)
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())

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
            ("deviceAssetId", str(uuid.uuid4())),
            ("deviceId", "comfyui-immich"),
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

    def _build_auto_description(self, prompt):
        """Build a description from the ComfyUI workflow prompt data."""
        if not prompt:
            return ""

        lines = []
        nodes = prompt if isinstance(prompt, dict) else {}

        # Find key nodes by class type
        positive_text = ""
        negative_text = ""
        checkpoint = ""
        seed = ""
        sampler = ""
        steps = ""
        cfg = ""

        for node in nodes.values():
            class_type = node.get("class_type", "")
            inputs = node.get("inputs", {})
            meta_title = node.get("_meta", {}).get("title", "")

            if class_type == "CLIPTextEncode":
                text = inputs.get("text", "")
                is_positive = meta_title in ("@prompt", "@positive") or (
                    not positive_text and meta_title not in ("@negative", "Hand Positive Prompt")
                )
                if is_positive and text and meta_title != "Hand Positive Prompt":
                    positive_text = positive_text or text
                if meta_title == "@negative":
                    negative_text = text

            elif class_type == "CheckpointLoaderSimple":
                checkpoint = inputs.get("ckpt_name", "")

            elif class_type == "KSampler":
                seed = str(inputs.get("seed", ""))
                sampler = inputs.get("sampler_name", "")
                steps = str(inputs.get("steps", ""))
                cfg = str(inputs.get("cfg", ""))

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
        description="",
        album_id="",
        filename_prefix="ComfyUI",
        prompt=None,
        extra_pnginfo=None,
    ):
        immich_url, api_key = self._get_config()

        # Auto-build description from prompt data when none provided
        if not description and prompt:
            description = self._build_auto_description(prompt)

        results = []
        batch_size = images.shape[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"[SaveToImmich] Uploading {batch_size} image(s) to {immich_url}")

        for i in range(batch_size):
            try:
                # Convert tensor to PNG with full embedded metadata
                png_bytes = self._build_png_bytes(
                    images[i], prompt=prompt, extra_pnginfo=extra_pnginfo
                )

                filename = f"{filename_prefix}_{timestamp}_{i:04d}.png"

                # Upload with metadata already embedded in the PNG
                asset_id = self._upload_asset(immich_url, api_key, png_bytes, filename)
                if not asset_id:
                    print(
                        f"[SaveToImmich] WARNING: Upload {i + 1}/{batch_size} returned no asset ID"
                    )
                    continue

                print(f"[SaveToImmich] Uploaded {i + 1}/{batch_size}: {filename} -> {asset_id}")

                # Set Immich description (visible in the UI)
                if description:
                    try:
                        self._set_description(immich_url, api_key, asset_id, description)
                    except (HTTPError, URLError) as e:
                        print(f"[SaveToImmich] WARNING: Failed to set description: {e}")

                # Add to album
                if album_id:
                    try:
                        self._add_to_album(immich_url, api_key, album_id, asset_id)
                    except (HTTPError, URLError) as e:
                        print(f"[SaveToImmich] WARNING: Failed to add to album: {e}")

                results.append({"asset_id": asset_id, "filename": filename})

            except (HTTPError, URLError) as e:
                print(f"[SaveToImmich] ERROR: Failed to upload image {i + 1}/{batch_size}: {e}")
            except Exception as e:
                print(f"[SaveToImmich] ERROR: Unexpected error on image {i + 1}/{batch_size}: {e}")

        print(f"[SaveToImmich] Done. {len(results)}/{batch_size} image(s) uploaded successfully.")
        return {"ui": {"images": results}}
