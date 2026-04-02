"""Tests for SaveToImmich node."""

import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from nodes.save_to_immich import SaveToImmich, _load_env, _multipart_encode

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


# --- SaveToImmich ---


class TestSaveToImmich:
    def test_input_types_structure(self):
        inputs = SaveToImmich.INPUT_TYPES()
        assert "images" in inputs["required"]
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
            patch("nodes.save_to_immich.os.path.dirname", return_value=str(tmp_path)),
            pytest.raises(ValueError, match="IMMICH_URL not set"),
        ):
            node._get_config()

    @patch("nodes.save_to_immich.urlopen")
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

        with patch.object(node, "_get_config", return_value=("https://immich.test", "test-key")):
            result = node.upload(
                mock_images,
                description="test image",
                filename_prefix="test",
            )

        assert len(result["ui"]["images"]) == 1
        assert result["ui"]["images"][0]["asset_id"] == "asset-123"
