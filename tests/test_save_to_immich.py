"""Tests for SaveToImmich node."""

import contextlib
import http.client
import io
import json
import socket
import sys
import threading
import time
import urllib.request
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
from urllib.request import Request

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from immich_nodes.save_to_immich import (
    SaveToImmich,
    _load_env,
    _multipart_encode,
    _normalize_immich_url,
    _print_batch_summary,
)

# --- _load_env ---


class TestLoadEnv:
    def test_parses_simple_env(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("IMMICH_URL=https://immich.test\nIMMICH_API_KEY=abc123\n")
        result = _load_env(str(env_file))
        assert result["IMMICH_URL"] == "https://immich.test"
        assert result["IMMICH_API_KEY"] == "abc123"

    def test_strips_quotes(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text('IMMICH_URL="https://immich.test"\n')
        result = _load_env(str(env_file))
        assert result["IMMICH_URL"] == "https://immich.test"

    def test_skips_comments_and_blanks(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nIMMICH_URL=test\n")
        result = _load_env(str(env_file))
        assert len(result) == 1
        assert result["IMMICH_URL"] == "test"

    def test_missing_file_returns_empty(self):
        result = _load_env("/nonexistent/.env")
        assert result == {}


# --- _multipart_encode ---


class TestMultipartEncode:
    def test_encodes_fields_and_files(self):
        fields = [("key", "value")]
        files = [("file", "test.png", "image/png", b"\x89PNG")]
        body, content_type = _multipart_encode(fields, files)
        assert b"key" in body
        assert b"value" in body
        assert b"test.png" in body
        assert b"\x89PNG" in body
        assert "multipart/form-data; boundary=" in content_type


class TestNormalizeImmichUrl:
    def test_strips_trailing_slash(self):
        assert _normalize_immich_url("https://immich.test/") == "https://immich.test"

    def test_accepts_api_url(self):
        assert _normalize_immich_url("https://immich.test/api/") == "https://immich.test"


# --- SaveToImmich ---


class TestSaveToImmich:
    def test_input_types_structure(self):
        inputs = SaveToImmich.INPUT_TYPES()
        assert "images" in inputs["required"]
        assert "character" in inputs["optional"]
        assert "description" in inputs["optional"]
        assert "album_id" in inputs["optional"]
        assert "filename_prefix" in inputs["optional"]
        assert "prompt" in inputs["hidden"]
        assert "extra_pnginfo" in inputs["hidden"]

    def test_node_properties(self):
        assert SaveToImmich.OUTPUT_NODE is True
        assert SaveToImmich.RETURN_TYPES == ()
        assert SaveToImmich.FUNCTION == "upload"
        assert SaveToImmich.CATEGORY == "image/immich"

    def test_build_png_bytes_embeds_metadata(self):
        node = SaveToImmich()
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value = mock_tensor
        mock_tensor.numpy.return_value = np.random.rand(64, 64, 3).astype(np.float32)

        prompt = {"1": {"class_type": "KSampler", "inputs": {"seed": 42}}}
        extra = {"workflow": {"nodes": []}}

        png_bytes = node._build_png_bytes(mock_tensor, prompt=prompt, extra_pnginfo=extra)

        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (64, 64)
        assert "prompt" in img.info
        assert json.loads(img.info["prompt"]) == prompt
        assert "workflow" in img.info
        assert json.loads(img.info["workflow"]) == {"nodes": []}

    def test_build_png_bytes_without_metadata(self):
        node = SaveToImmich()
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value = mock_tensor
        mock_tensor.numpy.return_value = np.random.rand(32, 32, 3).astype(np.float32)

        png_bytes = node._build_png_bytes(mock_tensor)

        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (32, 32)

    def test_get_config_raises_without_env(self, tmp_path):
        node = SaveToImmich()
        with (
            patch("immich_nodes.save_to_immich.os.path.dirname", return_value=str(tmp_path)),
            pytest.raises(ValueError, match="IMMICH_URL not set"),
        ):
            node._get_config()

    def test_get_config_allows_environment_override(self, tmp_path):
        node = SaveToImmich()
        env_file = tmp_path / ".env"
        env_file.write_text("IMMICH_URL=https://from-file.test\nIMMICH_API_KEY=file-key\n")

        with (
            patch("immich_nodes.save_to_immich.os.path.dirname", return_value=str(tmp_path)),
            patch.dict(
                "immich_nodes.save_to_immich.os.environ",
                {
                    "IMMICH_URL": "https://from-env.test/api/",
                    "IMMICH_API_KEY": " env-key ",
                },
            ),
        ):
            assert node._get_config() == ("https://from-env.test", "env-key")

    @patch("immich_nodes.save_to_immich.urlopen")
    def test_api_request_allows_empty_json_response(self, mock_urlopen):
        node = SaveToImmich()
        empty_resp = MagicMock()
        empty_resp.read.return_value = b""
        empty_resp.__enter__ = lambda s: s
        empty_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = empty_resp

        result = node._api_request("https://immich.test/api/no-content", "PUT", {})

        assert result == {}

    @patch("immich_nodes.save_to_immich.urlopen")
    def test_api_request_html_body_becomes_stage_failure(self, mock_urlopen):
        """A reverse proxy returning HTML must surface as a ValueError, not crash."""
        node = SaveToImmich()
        html_resp = MagicMock()
        html_resp.read.return_value = b"<html>502 Bad Gateway</html>"
        html_resp.__enter__ = lambda s: s
        html_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = html_resp

        with pytest.raises(ValueError, match="Expecting value"):
            node._api_request("https://immich.test/api/assets", "GET", {})

    def test_html_body_during_upload_lands_as_upload_stage_failure(self, image_batch):
        """End-to-end: an HTML 200 becomes a receipt, not a crash."""
        node = SaveToImmich()
        with (
            patch.object(node, "_get_config", return_value=("http://example.invalid", "key")),
            patch.object(node, "_build_png_bytes", return_value=b"png"),
            patch.object(node, "_save_comfy_preview", return_value={"filename": "kept.png"}),
            patch.object(node, "_api_request") as api,
        ):
            api.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
            result = node.upload(image_batch)

        report = result["ui"]["archive"][0]
        assert report["upload"] == "failed"
        assert report["errors"][0]["stage"] == "upload"

    def test_save_comfy_preview_writes_output_file(self, tmp_path):
        node = SaveToImmich()
        fake_folder_paths = MagicMock()
        fake_folder_paths.get_output_directory.return_value = str(tmp_path)

        with patch("immich_nodes.save_to_immich.folder_paths", fake_folder_paths):
            result = node._save_comfy_preview(b"png-bytes", "preview.png")

        assert result == {"filename": "preview.png", "subfolder": "", "type": "output"}
        assert (tmp_path / "preview.png").read_bytes() == b"png-bytes"

    @patch("immich_nodes.save_to_immich.urlopen")
    def test_upload_success(self, mock_urlopen, image_batch):
        """Full upload flow with mocked HTTP."""
        node = SaveToImmich()

        # Mock HTTP responses
        upload_resp = MagicMock()
        upload_resp.read.return_value = json.dumps({"id": "asset-123"}).encode()
        upload_resp.__enter__ = lambda s: s
        upload_resp.__exit__ = MagicMock(return_value=False)

        desc_resp = MagicMock()
        desc_resp.read.return_value = json.dumps({}).encode()
        desc_resp.__enter__ = lambda s: s
        desc_resp.__exit__ = MagicMock(return_value=False)

        settle_resp = MagicMock()
        settle_resp.read.return_value = json.dumps(
            {"id": "asset-123", "originalPath": "/data/library/admin/2026/x.png"}
        ).encode()
        settle_resp.__enter__ = lambda s: s
        settle_resp.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [upload_resp, settle_resp, desc_resp]

        with (
            patch.object(node, "_get_config", return_value=("https://immich.test", "test-key")),
            patch.object(
                node,
                "_save_comfy_preview",
                return_value={"filename": "preview.png", "subfolder": "", "type": "output"},
            ),
        ):
            result = node.upload(
                image_batch,
                description="test image",
                filename_prefix="test",
            )

        assert len(result["ui"]["images"]) == 1
        assert result["ui"]["images"][0]["asset_id"] == "asset-123"
        assert result["ui"]["images"][0]["filename"] == "preview.png"
        assert result["ui"]["images"][0]["subfolder"] == ""
        assert result["ui"]["images"][0]["type"] == "output"

        upload_request = mock_urlopen.call_args_list[0].args[0]
        assert upload_request.full_url == "https://immich.test/api/assets"
        assert b"fileCreatedAt" in upload_request.data
        assert b"fileModifiedAt" in upload_request.data
        assert b"assetData" in upload_request.data
        assert b"deviceAssetId" not in upload_request.data
        assert b"deviceId" not in upload_request.data

    def test_auto_description_reads_atelier_graph(self):
        node = SaveToImmich()
        prompt = {
            "101": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "moodyKrea2Mix_v60.safetensors"},
                "_meta": {"title": "Diffusion model"},
            },
            "201": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "### CHARACTER\nAoi\n\n### SCENE\nreading"},
                "_meta": {"title": "Positive prompt"},
            },
            "302": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": 42,
                    "sampler_name": "euler_ancestral",
                    "steps": 9,
                    "cfg": 1.0,
                },
                "_meta": {"title": "Sampler"},
            },
        }
        text = node._build_auto_description(prompt, character="Aoi Katsuragi")
        assert text.startswith("Character: Aoi Katsuragi")
        assert "Checkpoint: moodyKrea2Mix_v60.safetensors" in text
        assert "Sampler: euler_ancestral | Steps: 9 | CFG: 1.0" in text
        assert "Seed: 42" in text
        assert "### CHARACTER\nAoi" in text

    def test_auto_description_reads_atelier_at_prompt_title(self):
        node = SaveToImmich()
        prompt = {
            "101": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "redcraftMinimaxH3REDMIX_30Krea2.safetensors"},
                "_meta": {"title": "@model"},
            },
            "301": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "a quiet kitchen"},
                "_meta": {"title": "@prompt"},
            },
        }
        text = node._build_auto_description(prompt)
        assert "Checkpoint: redcraftMinimaxH3REDMIX_30Krea2.safetensors" in text
        assert "Positive: a quiet kitchen" in text

    def test_auto_description_reads_checkpoint_loader(self):
        node = SaveToImmich()
        prompt = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "old.safetensors"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "a kitchen"},
                "_meta": {"title": "@positive"},
            },
        }
        text = node._build_auto_description(prompt)
        assert "Checkpoint: old.safetensors" in text
        assert "Positive: a kitchen" in text

    @patch("immich_nodes.save_to_immich.urlopen")
    def test_preview_is_kept_when_upload_fails(self, mock_urlopen, image_batch):
        node = SaveToImmich()
        mock_urlopen.side_effect = URLError("immich down")

        with (
            patch.object(node, "_get_config", return_value=("https://immich.test", "test-key")),
            patch.object(
                node,
                "_save_comfy_preview",
                return_value={"filename": "kept.png", "subfolder": "", "type": "output"},
            ) as preview,
        ):
            result = node.upload(image_batch, filename_prefix="test")

        preview.assert_called_once()
        assert len(result["ui"]["images"]) == 1
        assert result["ui"]["images"][0]["filename"] == "kept.png"
        assert "asset_id" not in result["ui"]["images"][0]


class TestStorageSettle:
    """The description PUT must not race Immich's storage template move.

    Immich queues the sidecar write from the description PUT and resolves the
    asset path as it stood when the job was queued. Setting a description while
    the file is still in upload/ leaves that job stat-ing a path the storage
    template has already moved, which fails as ENOENT and loses the .xmp.
    """

    def _asset_resp(self, path):
        resp = MagicMock()
        resp.read.return_value = json.dumps({"id": "asset-123", "originalPath": path}).encode()
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    @patch("immich_nodes.save_to_immich.time.sleep")
    @patch("immich_nodes.save_to_immich.urlopen")
    def test_waits_until_asset_leaves_upload(self, mock_urlopen, mock_sleep):
        node = SaveToImmich()
        mock_urlopen.side_effect = [
            self._asset_resp("/data/upload/user/ab/cd/asset-123.png"),
            self._asset_resp("/data/upload/user/ab/cd/asset-123.png"),
            self._asset_resp("/data/library/admin/2026/2026-08-19/render.png"),
        ]

        assert node._wait_for_storage_settle("https://immich.test", "key", "asset-123") is True
        assert mock_urlopen.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("immich_nodes.save_to_immich._SETTLE_TIMEOUT_SECONDS", 0.05)
    @patch("immich_nodes.save_to_immich.time.sleep")
    @patch("immich_nodes.save_to_immich.urlopen")
    def test_gives_up_when_asset_never_moves(self, mock_urlopen, mock_sleep):
        """Storage template engine off: the file stays in upload/ legitimately."""
        node = SaveToImmich()
        mock_urlopen.side_effect = lambda *a, **k: self._asset_resp(
            "/data/upload/user/ab/cd/asset-123.png"
        )

        assert node._wait_for_storage_settle("https://immich.test", "key", "asset-123") is False

    @patch("immich_nodes.save_to_immich.time.sleep")
    @patch("immich_nodes.save_to_immich.urlopen")
    def test_transient_lookup_error_keeps_polling(self, mock_urlopen, mock_sleep):
        """A failed GET says nothing about where the file is.

        Treating it as settled would fail open into the exact race this wait
        exists to close, so a transient error must not end the wait early.
        """
        node = SaveToImmich()
        mock_urlopen.side_effect = [
            URLError("boom"),
            self._asset_resp("/data/upload/user/ab/cd/asset-123.png"),
            self._asset_resp("/data/library/admin/2026/2026-08-19/render.png"),
        ]

        assert node._wait_for_storage_settle("https://immich.test", "key", "asset-123") is True
        assert mock_urlopen.call_count == 3

    @patch("immich_nodes.save_to_immich._SETTLE_TIMEOUT_SECONDS", 0.05)
    @patch("immich_nodes.save_to_immich.time.sleep")
    @patch("immich_nodes.save_to_immich.urlopen")
    def test_persistent_lookup_error_gives_up_at_deadline(self, mock_urlopen, mock_sleep):
        node = SaveToImmich()
        mock_urlopen.side_effect = URLError("boom")

        assert node._wait_for_storage_settle("https://immich.test", "key", "asset-123") is False
        assert mock_urlopen.call_count > 1

    @patch("immich_nodes.save_to_immich.time.sleep")
    @patch("immich_nodes.save_to_immich.urlopen")
    def test_socket_timeout_during_settle_is_treated_as_transient(self, mock_urlopen, mock_sleep):
        """A socket timeout mid-poll is not evidence the file has settled."""
        node = SaveToImmich()
        mock_urlopen.side_effect = [
            TimeoutError("read timeout"),
            self._asset_resp("/data/upload/user/ab/cd/asset-123.png"),
            self._asset_resp("/data/library/admin/2026/2026-08-19/render.png"),
        ]

        assert node._wait_for_storage_settle("https://immich.test", "key", "asset-123") is True
        assert mock_urlopen.call_count == 3

    @patch("immich_nodes.save_to_immich._SETTLE_TIMEOUT_SECONDS", 0.05)
    @patch("immich_nodes.save_to_immich.time.sleep")
    @patch("immich_nodes.save_to_immich.urlopen")
    def test_description_is_still_set_when_settle_times_out(
        self, mock_urlopen, mock_sleep, image_batch
    ):
        """A stuck move must degrade to the old behaviour, not drop the caption."""
        node = SaveToImmich()

        upload_resp = MagicMock()
        upload_resp.read.return_value = json.dumps({"id": "asset-123"}).encode()
        upload_resp.__enter__ = lambda s: s
        upload_resp.__exit__ = MagicMock(return_value=False)

        desc_resp = MagicMock()
        desc_resp.read.return_value = json.dumps({}).encode()
        desc_resp.__enter__ = lambda s: s
        desc_resp.__exit__ = MagicMock(return_value=False)

        stuck = self._asset_resp("/data/upload/user/ab/cd/asset-123.png")

        def responses(*args, **kwargs):
            req = args[0]
            if req.full_url.endswith("/api/assets") and req.method == "POST":
                return upload_resp
            if req.method == "PUT":
                return desc_resp
            return stuck

        mock_urlopen.side_effect = responses

        with (
            patch.object(node, "_get_config", return_value=("https://immich.test", "test-key")),
            patch.object(node, "_save_comfy_preview", return_value={"filename": "p.png"}),
        ):
            result = node.upload(image_batch, description="caption", filename_prefix="test")

        assert result["ui"]["images"][0]["asset_id"] == "asset-123"
        put_calls = [c for c in mock_urlopen.call_args_list if c.args[0].method == "PUT"]
        assert len(put_calls) == 1
        assert b"caption" in put_calls[0].args[0].data


@pytest.mark.parametrize(
    "filename",
    [
        "../escape.png",
        "/tmp/escape.png",
        "..\\escape.png",
        "C:\\escape.png",
        'bad"name.png',
        "bad\nname.png",
    ],
)
def test_preview_rejects_escaping_paths(tmp_path, filename):
    paths = MagicMock()
    paths.get_output_directory.return_value = str(tmp_path)
    with (
        patch("immich_nodes.save_to_immich.folder_paths", paths),
        pytest.raises(ValueError, match="output directory"),
    ):
        SaveToImmich()._save_comfy_preview(b"png", filename)
    assert list(tmp_path.iterdir()) == []


def test_preview_reports_safe_subfolder_and_refuses_symlink_escape(tmp_path):
    paths = MagicMock()
    paths.get_output_directory.return_value = str(tmp_path / "output")
    (tmp_path / "output").mkdir()
    (tmp_path / "output/escape").symlink_to(tmp_path, target_is_directory=True)
    with patch("immich_nodes.save_to_immich.folder_paths", paths):
        node = SaveToImmich()
        info = node._save_comfy_preview(b"png", "album/safe.png")
        assert info["filename"] == "safe.png" and info["subfolder"] == "album"
        with pytest.raises(ValueError, match="escapes"):
            node._save_comfy_preview(b"png", "escape/outside.png")
    assert not (tmp_path / "outside.png").exists()


def test_archive_failure_reports_stage_and_keeps_preview(image_batch):
    node = SaveToImmich()
    with (
        patch.object(node, "_get_config", side_effect=ValueError("missing configuration")),
        patch.object(node, "_build_png_bytes", return_value=b"png"),
        patch.object(node, "_save_comfy_preview", return_value={"filename": "kept.png"}),
    ):
        result = node.upload(image_batch)
    assert result["ui"]["images"][0]["filename"] == "kept.png"
    report = result["ui"]["archive"][0]
    assert report["upload"] == "failed"
    assert report["errors"][0]["stage"] == "upload"


def test_metadata_failure_does_not_hide_upload_or_skip_album():
    node = SaveToImmich()
    with (
        patch.object(node, "_get_config", return_value=("http://example.invalid", "key")),
        patch.object(node, "_upload_asset", return_value="asset-id"),
        patch.object(node, "_wait_for_storage_settle", return_value=True),
        patch.object(node, "_set_description", side_effect=TimeoutError("timeout")),
        patch.object(node, "_add_to_album") as album,
    ):
        report = node.archive_png(b"png", "image.png", description="caption", album_id="album")
    assert report["upload"] == "ok"
    assert report["asset_id"] == "asset-id"
    assert report["description"] == "failed"
    assert report["album"] == "ok"
    album.assert_called_once()


def test_retry_archive_preserves_png_bytes_and_can_resume_metadata(tmp_path):
    from immich_nodes.retry_archive import retry

    path = tmp_path / "original.png"
    metadata = PngInfo()
    metadata.add_text(
        "prompt",
        json.dumps(
            {
                "901": {
                    "class_type": "SaveToImmich",
                    "inputs": {"description": "exact caption", "album_id": "album-id"},
                }
            }
        ),
    )
    Image.new("RGB", (1, 1)).save(path, pnginfo=metadata)
    original = path.read_bytes()
    with patch.object(SaveToImmich, "archive_png", return_value={"errors": []}) as archive:
        assert retry(path, asset_id="existing") == {"errors": []}
    archive.assert_called_once_with(
        original,
        "original.png",
        description="exact caption",
        album_id="album-id",
        asset_id="existing",
    )
    assert path.read_bytes() == original


def test_existing_asset_id_never_reuploads_png():
    node = SaveToImmich()
    with (
        patch.object(node, "_get_config", return_value=("http://example.invalid", "key")),
        patch.object(node, "_upload_asset") as upload,
    ):
        report = node.archive_png(b"png", "image.png", asset_id="existing")
    upload.assert_not_called()
    assert report["upload"] == "reused" and report["asset_id"] == "existing"


@pytest.mark.parametrize(
    "result",
    [
        [{"id": "asset", "success": False, "error": "not_found"}],
    ],
)
def test_album_item_failure_is_not_http_success(result):
    """An entry that explicitly says success=False is a genuine failure."""
    node = SaveToImmich()
    with (
        patch.object(node, "_api_request", return_value=result),
        pytest.raises(ValueError, match="album membership"),
    ):
        node._add_to_album("https://immich.test", "key", "album", "asset")


@pytest.mark.parametrize(
    "result",
    [
        [],  # no per-asset entries at all
        [{"id": "other", "success": True}],  # entry for a different asset
        [{"id": "asset"}],  # entry for our asset but no `success` field
    ],
)
def test_album_ambiguous_response_is_unconfirmed_not_failed(result):
    """An entry that neither confirms nor explicitly fails is unconfirmed.

    Mirrors the empty-body case: Immich said nothing useful, so we cannot
    claim success or failure. The caller downgrades the receipt to
    `album: "unconfirmed"` rather than throwing.
    """
    node = SaveToImmich()
    with patch.object(node, "_api_request", return_value=result):
        assert node._add_to_album("https://immich.test", "key", "album", "asset") is False


@pytest.mark.parametrize("result", [{}, None, ""])
def test_album_add_without_per_asset_body_is_unconfirmed_not_failed(result):
    """A 204 or a body-stripping proxy carries no failure to report."""
    node = SaveToImmich()
    with patch.object(node, "_api_request", return_value=result):
        assert node._add_to_album("https://immich.test", "key", "album", "asset") is False
    with (
        patch.object(node, "_get_config", return_value=("http://example.invalid", "key")),
        patch.object(node, "_upload_asset", return_value="asset-id"),
        patch.object(node, "_api_request", return_value=result),
    ):
        report = node.archive_png(b"png", "image.png", album_id="album")
    assert report["album"] == "unconfirmed"
    assert report["errors"] == []


def test_album_add_returns_confirmation():
    node = SaveToImmich()
    with patch.object(node, "_api_request", return_value=[{"id": "asset", "success": True}]):
        assert node._add_to_album("https://immich.test", "key", "album", "asset") is True


@pytest.mark.parametrize(
    "item",
    [{"id": "asset", "success": True}, {"id": "asset", "success": False, "error": "duplicate"}],
)
def test_album_membership_accepts_insert_or_existing_member(item):
    node = SaveToImmich()
    with patch.object(node, "_api_request", return_value=[item]):
        node._add_to_album("https://immich.test", "key", "album", "asset")


def test_one_bad_image_keeps_the_rest_of_the_batch_and_its_receipts(image_batch_of_three):
    """An encode failure mid-batch must not strand asset IDs already in Immich."""
    node = SaveToImmich()
    images = image_batch_of_three
    encoded = {"n": 0}

    def encode(tensor, prompt=None, extra_pnginfo=None):
        encoded["n"] += 1
        if encoded["n"] == 2:
            raise OSError("no space left on device")
        return b"png"

    with (
        patch.object(node, "_get_config", return_value=("http://example.invalid", "key")),
        patch.object(node, "_build_png_bytes", side_effect=encode),
        patch.object(node, "_save_comfy_preview", side_effect=lambda b, f: {"filename": f}),
        patch.object(node, "_upload_asset", side_effect=["asset-1", "asset-3"]),
    ):
        result = node.upload(images)

    reports = result["ui"]["archive"]
    assert [r["upload"] for r in reports] == ["ok", "failed", "ok"]
    assert [r.get("asset_id") for r in reports] == ["asset-1", None, "asset-3"]
    assert reports[1]["errors"][0]["stage"] == "encode"
    assert len(result["ui"]["images"]) == 2


def test_settle_timeout_does_not_disable_checks_for_later_assets(image_batch_of_two):
    """A timeout belongs to one asset; later assets still need the sidecar guard."""
    node = SaveToImmich()
    settle_requests = []

    def archive(image, filename, **kwargs):
        settle_requests.append(kwargs["wait_for_settle"])
        return (
            {
                "filename": filename,
                "upload": "ok",
                "description": "ok",
                "album": "not_requested",
                "storage_settled": len(settle_requests) != 1,
                "errors": [],
            },
            None,
        )

    with patch.object(node, "_archive_image", side_effect=archive):
        node.upload(image_batch_of_two, description="caption")

    assert settle_requests == [True, True]


def test_rejected_filename_prefix_still_archives_to_immich(image_batch):
    """Preview delivery and archiving fail independently, in both directions."""
    node = SaveToImmich()
    images = image_batch
    with (
        patch.object(node, "_get_config", return_value=("http://example.invalid", "key")),
        patch.object(node, "_build_png_bytes", return_value=b"png"),
        patch.object(
            node,
            "_save_comfy_preview",
            side_effect=ValueError("filename_prefix escapes the ComfyUI output directory"),
        ),
        patch.object(node, "_upload_asset", return_value="asset-id"),
    ):
        result = node.upload(images, filename_prefix="../escape")

    report = result["ui"]["archive"][0]
    assert result["ui"]["images"] == []
    assert report["upload"] == "ok" and report["asset_id"] == "asset-id"
    assert report["errors"][0]["stage"] == "preview"


def test_node_bugs_are_not_disguised_as_stage_failures():
    """A TypeError in our own code must crash, not surface as a tidy receipt."""
    node = SaveToImmich()
    with (
        patch.object(node, "_get_config", return_value=("http://example.invalid", "key")),
        patch.object(node, "_upload_asset", side_effect=TypeError("bug in this node")),
        pytest.raises(TypeError),
    ):
        node.archive_png(b"png", "image.png")


@pytest.mark.parametrize(
    ("chunk", "message"),
    [
        (None, "nothing to recover"),
        ("not json", "not valid JSON"),
        (json.dumps([{"class_type": "SaveToImmich"}]), "mapping of nodes"),
    ],
)
def test_retry_refuses_a_png_without_a_usable_graph(tmp_path, chunk, message):
    from immich_nodes.retry_archive import retry

    path = tmp_path / "original.png"
    metadata = PngInfo()
    if chunk is not None:
        metadata.add_text("prompt", chunk)
    Image.new("RGB", (1, 1)).save(path, pnginfo=metadata)

    with (
        patch.object(SaveToImmich, "archive_png") as archive,
        pytest.raises(ValueError, match=message),
    ):
        retry(path)
    archive.assert_not_called()


def test_retry_with_asset_id_refuses_when_there_is_nothing_left_to_do(tmp_path):
    from immich_nodes.retry_archive import retry

    path = tmp_path / "original.png"
    metadata = PngInfo()
    metadata.add_text("prompt", json.dumps({"1": {"class_type": "PreviewImage", "inputs": {}}}))
    Image.new("RGB", (1, 1)).save(path, pnginfo=metadata)

    with pytest.raises(ValueError, match="nothing to retry"):
        retry(path, asset_id="existing")


def test_retry_cli_reports_input_errors_without_a_traceback(tmp_path, capsys):
    from immich_nodes import retry_archive

    path = tmp_path / "plain.png"
    Image.new("RGB", (1, 1)).save(path)
    with patch.object(sys, "argv", ["retry_archive", str(path)]):
        assert retry_archive.main() == 1
    assert "nothing to recover" in capsys.readouterr().err


def test_an_unexpected_bug_still_leaves_receipts_for_the_rest_of_the_batch(image_batch_of_two):
    """Containment is unconditional: even a TypeError becomes a receipt."""
    node = SaveToImmich()
    images = image_batch_of_two
    with (
        patch.object(node, "_get_config", return_value=("http://example.invalid", "key")),
        patch.object(node, "_build_png_bytes", return_value=b"png"),
        patch.object(node, "_save_comfy_preview", side_effect=lambda b, f: {"filename": f}),
        patch.object(node, "_upload_asset", side_effect=[TypeError("bug in this node"), "asset-2"]),
    ):
        result = node.upload(images)

    reports = result["ui"]["archive"]
    assert [r["upload"] for r in reports] == ["failed", "ok"]
    assert reports[0]["errors"][0]["stage"] == "unexpected"
    assert "TypeError" in reports[0]["errors"][0]["message"]
    assert reports[1]["asset_id"] == "asset-2"


def test_a_working_archive_is_not_silent(capsys, image_batch_of_two):
    """Silence is indistinguishable from a node that never ran."""
    node = SaveToImmich()
    images = image_batch_of_two
    with (
        patch.object(node, "_get_config", return_value=("http://example.invalid", "key")),
        patch.object(node, "_build_png_bytes", return_value=b"png"),
        patch.object(node, "_save_comfy_preview", side_effect=lambda b, f: {"filename": f}),
        patch.object(node, "_upload_asset", side_effect=["asset-1", "asset-2"]),
    ):
        node.upload(images)
    assert "Archived 2/2 image(s)" in capsys.readouterr().out


def test_unconfirmed_album_is_explained_on_the_console(capsys, image_batch):
    """The state exists to be read; a receipt nobody sees is worthless."""
    node = SaveToImmich()
    images = image_batch
    with (
        patch.object(node, "_get_config", return_value=("http://example.invalid", "key")),
        patch.object(node, "_build_png_bytes", return_value=b"png"),
        patch.object(node, "_save_comfy_preview", side_effect=lambda b, f: {"filename": f}),
        patch.object(node, "_upload_asset", return_value="asset-1"),
        patch.object(node, "_api_request", return_value={}),
    ):
        result = node.upload(images, album_id="album")

    out = capsys.readouterr().out
    assert result["ui"]["archive"][0]["album"] == "unconfirmed"
    assert "without returning a per-asset confirmation" in out
    assert "reverse proxy" in out
    assert "failed" not in out


def test_summary_counts_a_reused_asset_as_archived(capsys):
    node = SaveToImmich()
    with (
        patch.object(node, "_get_config", return_value=("http://example.invalid", "key")),
        patch.object(node, "_upload_asset") as upload,
    ):
        report = node.archive_png(b"png", "image.png", asset_id="existing")
    upload.assert_not_called()
    _print_batch_summary([report])
    assert "Archived 1/1 image(s)" in capsys.readouterr().out


def test_retry_cli_explains_an_unconfirmed_album(tmp_path, capsys):
    from immich_nodes import retry_archive

    path = tmp_path / "original.png"
    metadata = PngInfo()
    metadata.add_text(
        "prompt",
        json.dumps(
            {"1": {"class_type": "SaveToImmich", "inputs": {"description": "c", "album_id": "a"}}}
        ),
    )
    Image.new("RGB", (1, 1)).save(path, pnginfo=metadata)

    report = {"description": "ok", "album": "unconfirmed", "errors": []}
    with (
        patch.object(retry_archive, "retry", return_value=report),
        patch.object(sys, "argv", ["retry_archive", str(path)]),
    ):
        assert retry_archive.main() == 0
    assert "proxy is stripping response bodies" in capsys.readouterr().err


def test_failure_names_the_file_and_points_at_the_recovery_tool(capsys, image_batch_of_two):
    """The named file is exactly the argument retry_archive takes."""
    node = SaveToImmich()
    images = image_batch_of_two
    with (
        patch.object(node, "_get_config", return_value=("http://example.invalid", "key")),
        patch.object(node, "_build_png_bytes", return_value=b"png"),
        patch.object(
            node, "_save_comfy_preview", side_effect=lambda b, f: {"filename": f, "subfolder": ""}
        ),
        patch.object(node, "_upload_asset", side_effect=[URLError("refused"), "asset-2"]),
    ):
        result = node.upload(images, filename_prefix="shot")

    out = capsys.readouterr().out
    failed = result["ui"]["archive"][0]["filename"]
    assert f"{failed}: upload failed" in out
    assert "1 image(s) did not reach Immich" in out
    assert "retry_archive" in out


def test_no_retry_hint_when_there_is_no_local_png_to_retry_from(capsys, image_batch):
    """Preview and upload both failed: nothing on disk, so the hint would lie."""
    node = SaveToImmich()
    images = image_batch
    with (
        patch.object(node, "_get_config", return_value=("http://example.invalid", "key")),
        patch.object(node, "_build_png_bytes", return_value=b"png"),
        patch.object(node, "_save_comfy_preview", side_effect=ValueError("bad prefix")),
        patch.object(node, "_upload_asset", side_effect=URLError("refused")),
    ):
        node.upload(images)

    out = capsys.readouterr().out
    assert "1 image(s) did not reach Immich" in out
    assert "retry_archive" not in out


def _first_bytes_from(send):
    """Run `send(port)` against a local listener and return the first bytes it received."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]
    received = []

    def accept():
        conn, _ = server.accept()
        conn.settimeout(3)
        try:
            received.append(conn.recv(5))
        finally:
            conn.close()

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    with contextlib.suppress(Exception):
        send(port)
    thread.join(timeout=5)
    server.close()
    return received[0] if received else b""


class TestNoProcessWideSideEffects:
    """Importing the node must not change networking for the rest of ComfyUI."""

    def test_import_leaves_http_client_untouched(self):
        from immich_nodes import save_to_immich  # noqa: F401

        assert http.client.HTTPConnection.connect.__module__ == "http.client"
        assert http.client.HTTPSConnection.connect.__module__ == "http.client"

    def test_import_installs_no_global_opener(self):
        from immich_nodes import save_to_immich  # noqa: F401

        opener = urllib.request._opener
        headers = dict(opener.addheaders) if opener is not None else {}
        assert not headers.get("User-Agent", "").startswith("comfyui-immich")

    def test_https_subclass_still_does_tls_after_import(self):
        """Regression: the old global patch sent this subclass down plain HTTP."""
        from immich_nodes import save_to_immich  # noqa: F401

        class ThirdPartyHTTPS(http.client.HTTPSConnection):
            pass  # no connect() of its own

        def send(port):
            conn = ThirdPartyHTTPS("127.0.0.1", port, timeout=3)
            conn.request("GET", "/", headers={"Authorization": "Bearer not-a-real-token"})

        first = _first_bytes_from(send)
        assert first[:1] == b"\x16", f"expected a TLS handshake, got {first!r}"


class TestScopedTimeouts:
    """Immich calls go through this module's own bounded connections only."""

    def test_private_opener_refuses_redirects_and_keeps_stdlib_transport(self):
        from immich_nodes import save_to_immich as m

        kinds = {type(h) for h in m._OPENER.handlers}
        assert m._RefuseRedirects in kinds
        assert urllib.request.HTTPRedirectHandler not in kinds
        # Plain stdlib transport: TLS and certificate checks are Python's own.
        assert urllib.request.HTTPHandler in kinds
        assert urllib.request.HTTPSHandler in kinds
        assert dict(m._OPENER.addheaders)["User-Agent"].startswith("comfyui-immich/")

    @pytest.mark.parametrize("chunked", [False, True], ids=["content-length", "chunked"])
    def test_a_server_that_stalls_mid_body_times_out(self, chunked):
        """The timeout passed to urlopen bounds reads, not only connect.

        The server sends headers and part of the body, then goes silent. Before
        this test, custom connection classes re-pinned the timeout to guard
        exactly this case; plain urllib already raises, so they were removed.
        """
        from immich_nodes import save_to_immich as m

        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        release = threading.Event()

        def serve():
            conn, _ = server.accept()
            conn.recv(65536)
            framing = (
                b"Transfer-Encoding: chunked\r\n\r\n3e8\r\n"
                if chunked
                else b"Content-Length: 100000\r\n\r\n"
            )
            conn.sendall(b"HTTP/1.1 200 OK\r\n" + framing + b"x" * 1000)
            release.wait(10)
            conn.close()
            server.close()

        threading.Thread(target=serve, daemon=True).start()
        started = time.monotonic()
        try:
            with (
                pytest.raises(TimeoutError),
                m.urlopen(Request(f"http://127.0.0.1:{port}/"), timeout=0.5) as resp,
            ):
                resp.read(100)
                resp.read()
        finally:
            release.set()
        assert time.monotonic() - started < 5

    def test_our_https_requests_still_do_tls(self):
        from immich_nodes import save_to_immich as m

        def send(port):
            m.urlopen(Request(f"https://127.0.0.1:{port}/"), timeout=3)

        first = _first_bytes_from(send)
        assert first[:1] == b"\x16", f"expected a TLS handshake, got {first!r}"

    def test_our_http_requests_carry_user_agent(self):
        from immich_nodes import save_to_immich as m

        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        seen = {}

        def serve():
            conn, _ = server.accept()
            seen["request"] = conn.recv(4096)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
            conn.close()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        with m.urlopen(Request(f"http://127.0.0.1:{port}/api"), timeout=3) as resp:
            assert resp.read() == b"{}"
        thread.join(timeout=5)
        server.close()
        assert b"User-Agent: comfyui-immich/" in seen["request"]

    def test_request_timeout_is_a_finite_positive_number(self):
        from immich_nodes import save_to_immich as m

        assert m._REQUEST_TIMEOUT_SECONDS > 0
        assert m._REQUEST_TIMEOUT_SECONDS <= 120  # not absurdly long


class TestRedirectsAreRefused:
    """A redirect must never carry the API key to a second request."""

    @staticmethod
    def _server(handle):
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(2)
        port = server.getsockname()[1]
        log = []

        def run():
            server.settimeout(3)
            with contextlib.suppress(OSError):
                while True:
                    conn, _ = server.accept()
                    conn.settimeout(3)
                    data = conn.recv(4096)
                    log.append(data)
                    conn.sendall(handle(data))
                    conn.close()

        threading.Thread(target=run, daemon=True).start()
        return port, log, server

    def _redirect_to(self, target_port, same_host_path=False):
        location = f"http://127.0.0.1:{target_port}/" + ("moved" if same_host_path else "")
        reply = (
            f"HTTP/1.1 302 Found\r\nLocation: {location}\r\n"
            "Content-Length: 0\r\nConnection: close\r\n\r\n"
        ).encode()
        return lambda _data: reply

    def test_cross_host_redirect_sends_nothing_to_second_host(self):
        from immich_nodes import save_to_immich as m

        ok = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"
        other_port, other_log, other = self._server(lambda _d: ok)
        first_port, _first_log, first = self._server(self._redirect_to(other_port))
        req = Request(
            f"http://127.0.0.1:{first_port}/api/assets", headers={"x-api-key": "SENTINEL-KEY-7f3a"}
        )
        with pytest.raises(HTTPError):
            m.urlopen(req, timeout=3)
        first.close()
        other.close()
        assert other_log == [], "the redirect target must receive no request at all"

    def test_same_host_redirect_is_refused_too(self):
        from immich_nodes import save_to_immich as m

        port_holder = {}

        def handle(data):
            if b"GET /moved" in data:
                return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}"
            return self._redirect_to(port_holder["p"], same_host_path=True)(data)

        port, log, server = self._server(handle)
        port_holder["p"] = port
        with pytest.raises(HTTPError) as err:
            m.urlopen(
                Request(f"http://127.0.0.1:{port}/api", headers={"x-api-key": "k"}), timeout=3
            )
        server.close()
        assert err.value.code == 302
        assert "IMMICH_URL" in str(err.value.reason)
        assert len(log) == 1, "no second request after the redirect"
