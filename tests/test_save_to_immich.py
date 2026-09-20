"""Tests for SaveToImmich node."""

import io
import json
import sys
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from immich_nodes.save_to_immich import (
    SaveToImmich,
    _load_env,
    _multipart_encode,
    _normalize_immich_url,
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

    def test_save_comfy_preview_writes_output_file(self, tmp_path):
        node = SaveToImmich()
        fake_folder_paths = MagicMock()
        fake_folder_paths.get_output_directory.return_value = str(tmp_path)

        with patch("immich_nodes.save_to_immich.folder_paths", fake_folder_paths):
            result = node._save_comfy_preview(b"png-bytes", "preview.png")

        assert result == {"filename": "preview.png", "subfolder": "", "type": "output"}
        assert (tmp_path / "preview.png").read_bytes() == b"png-bytes"

    @patch("immich_nodes.save_to_immich.urlopen")
    def test_upload_success(self, mock_urlopen):
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

        # Create fake image tensor (batch of 1)
        mock_images = MagicMock()
        mock_images.shape = [1]
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value = mock_tensor
        mock_tensor.numpy.return_value = np.random.rand(32, 32, 3).astype(np.float32)
        mock_images.__getitem__ = lambda s, i: mock_tensor

        with (
            patch.object(node, "_get_config", return_value=("https://immich.test", "test-key")),
            patch.object(
                node,
                "_save_comfy_preview",
                return_value={"filename": "preview.png", "subfolder": "", "type": "output"},
            ),
        ):
            result = node.upload(
                mock_images,
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
    def test_preview_is_kept_when_upload_fails(self, mock_urlopen):
        node = SaveToImmich()
        mock_urlopen.side_effect = URLError("immich down")

        mock_images = MagicMock()
        mock_images.shape = [1]
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value = mock_tensor
        mock_tensor.numpy.return_value = np.random.rand(32, 32, 3).astype(np.float32)
        mock_images.__getitem__ = lambda s, i: mock_tensor

        with (
            patch.object(node, "_get_config", return_value=("https://immich.test", "test-key")),
            patch.object(
                node,
                "_save_comfy_preview",
                return_value={"filename": "kept.png", "subfolder": "", "type": "output"},
            ) as preview,
        ):
            result = node.upload(mock_images, filename_prefix="test")

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

    @patch("immich_nodes.save_to_immich._SETTLE_TIMEOUT_SECONDS", 0.05)
    @patch("immich_nodes.save_to_immich.time.sleep")
    @patch("immich_nodes.save_to_immich.urlopen")
    def test_description_is_still_set_when_settle_times_out(self, mock_urlopen, mock_sleep):
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

        mock_images = MagicMock()
        mock_images.shape = [1]
        mock_tensor = MagicMock()
        mock_tensor.cpu.return_value = mock_tensor
        mock_tensor.numpy.return_value = np.random.rand(32, 32, 3).astype(np.float32)
        mock_images.__getitem__ = lambda s, i: mock_tensor

        with (
            patch.object(node, "_get_config", return_value=("https://immich.test", "test-key")),
            patch.object(node, "_save_comfy_preview", return_value={"filename": "p.png"}),
        ):
            result = node.upload(mock_images, description="caption", filename_prefix="test")

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


def test_archive_failure_reports_stage_and_keeps_preview():
    node = SaveToImmich()
    images = MagicMock()
    images.shape = [1]
    with (
        patch.object(node, "_get_config", side_effect=ValueError("missing configuration")),
        patch.object(node, "_build_png_bytes", return_value=b"png"),
        patch.object(node, "_save_comfy_preview", return_value={"filename": "kept.png"}),
    ):
        result = node.upload(images)
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
        [],
        [{"id": "asset", "success": False, "error": "not_found"}],
        [{"id": "other", "success": True}],
    ],
)
def test_album_item_failure_is_not_http_success(result):
    node = SaveToImmich()
    with (
        patch.object(node, "_api_request", return_value=result),
        pytest.raises(ValueError, match="album membership"),
    ):
        node._add_to_album("https://immich.test", "key", "album", "asset")


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


def test_one_bad_image_keeps_the_rest_of_the_batch_and_its_receipts():
    """An encode failure mid-batch must not strand asset IDs already in Immich."""
    node = SaveToImmich()
    images = MagicMock()
    images.shape = [3]
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


def test_rejected_filename_prefix_still_archives_to_immich():
    """Preview delivery and archiving fail independently, in both directions."""
    node = SaveToImmich()
    images = MagicMock()
    images.shape = [1]
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


def test_an_unexpected_bug_still_leaves_receipts_for_the_rest_of_the_batch():
    """Containment is unconditional: even a TypeError becomes a receipt."""
    node = SaveToImmich()
    images = MagicMock()
    images.shape = [2]
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
