"""Tests for SaveToImmich node."""

import io
import json
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import numpy as np
import pytest
from PIL import Image

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

        mock_urlopen.side_effect = [upload_resp, desc_resp]

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
