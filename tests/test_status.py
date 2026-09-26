"""Status contract v1: read-only status, input-free rate-limited test, no key material."""

import json
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from immich_nodes import save_to_immich as node_mod
from immich_nodes import status

SENTINEL = "SENTINEL-kq7Z-KEY-93xw"
URL = "https://immich.example.test"


@pytest.fixture(autouse=True)
def _configured(tmp_path):
    status._last_test[0] = None
    user_env = tmp_path / "comfyui-immich.env"
    user_env.write_text(f"IMMICH_URL={URL}/api/\nIMMICH_API_KEY={SENTINEL}\n")
    with patch.object(
        node_mod, "_config_paths", return_value=(str(user_env), str(tmp_path / ".env"))
    ):
        yield
    status._last_test[0] = None


def _leaks(payload, key=SENTINEL):
    raw = json.dumps(payload)
    return any(fragment in raw for fragment in (key, key[:4], key[-4:]))


def _ok_response(code=200):
    resp = MagicMock()
    resp.status = code
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestNoKeyMaterial:
    def test_status_has_no_key_material(self):
        payload = status.status_payload()
        assert payload["key_set"] is True
        assert payload["url"] == URL
        assert not _leaks(payload)

    @pytest.mark.parametrize(
        "outcome",
        [_ok_response(), HTTPError(URL, 401, "no", {}, None), URLError("down")],
    )
    def test_test_results_have_no_key_material(self, outcome):
        with patch("immich_nodes.save_to_immich.urlopen") as mock_open:
            if isinstance(outcome, Exception):
                mock_open.side_effect = outcome
            else:
                mock_open.return_value = outcome
            _code, payload = status.run_connection_test(False)
        assert not _leaks(payload)

    def test_mutation_proof_detector_catches_a_leaky_status(self):
        leaky = dict(status.status_payload(), key=SENTINEL)
        assert _leaks(leaky)
        assert _leaks({"key_hint": SENTINEL[-4:]})


class TestTestRoute:
    def test_input_is_refused_and_nothing_is_sent(self):
        with patch("immich_nodes.save_to_immich.urlopen") as mock_open:
            code, payload = status.run_connection_test(True)
        assert code == 400
        assert payload["error"] == "input_not_accepted"
        mock_open.assert_not_called()

    def test_single_request_goes_to_saved_url_with_saved_key(self):
        with patch("immich_nodes.save_to_immich.urlopen", return_value=_ok_response()) as mock_open:
            code, payload = status.run_connection_test(False)
        assert (code, payload) == (200, {"ok": True, "status": 200, "error": None})
        assert mock_open.call_count == 1
        req = mock_open.call_args.args[0]
        assert req.full_url == URL + "/api/users/me"
        assert req.get_header("X-api-key") == SENTINEL

    def test_rate_limited_within_window(self):
        times = iter([100.0, 101.0, 106.0])
        with patch("immich_nodes.save_to_immich.urlopen", return_value=_ok_response()):
            first = status.run_connection_test(False, clock=lambda: next(times))
            second = status.run_connection_test(False, clock=lambda: next(times))
            third = status.run_connection_test(False, clock=lambda: next(times))
        assert first[0] == 200
        assert second == (429, {"ok": False, "status": None, "error": "rate_limited"})
        assert third[0] == 200

    @pytest.mark.parametrize(
        ("code", "error"),
        [
            (401, "unauthorized"),
            (403, "forbidden"),
            (302, "redirect_refused"),
            (500, "bad_response"),
        ],
    )
    def test_http_errors_become_classes(self, code, error):
        with patch(
            "immich_nodes.save_to_immich.urlopen", side_effect=HTTPError(URL, code, "x", {}, None)
        ):
            _code, payload = status.run_connection_test(False)
        assert payload == {"ok": False, "status": code, "error": error}

    def test_not_configured_makes_no_request(self, tmp_path):
        (tmp_path / "comfyui-immich.env").write_text(f"IMMICH_URL={URL}\n")
        with patch("immich_nodes.save_to_immich.urlopen") as mock_open:
            _code, payload = status.run_connection_test(False)
        assert payload["error"] == "not_configured"
        mock_open.assert_not_called()


class TestConfigPrecedence:
    def test_userdir_then_node_dotenv_and_environment_ignored(self, tmp_path):
        user_env = tmp_path / "comfyui-immich.env"
        node_env = tmp_path / ".env"
        user_env.write_text("IMMICH_URL=https://from-user.test\n")
        node_env.write_text("IMMICH_URL=https://from-node.test\nIMMICH_API_KEY=node-key\n")
        with (
            patch.object(node_mod, "_config_paths", return_value=(str(user_env), str(node_env))),
            patch.dict(
                "os.environ", {"IMMICH_URL": "https://from-env.test", "IMMICH_API_KEY": "env-key"}
            ),
        ):
            config = node_mod.resolve_config()
        assert (config["url"], config["url_source"]) == ("https://from-user.test", "userdir")
        # No key in the user file: the legacy node .env fills in, never the environment.
        assert (config["key"], config["key_source"]) == ("node-key", "dotenv")

    def test_node_dotenv_is_last_resort(self, tmp_path):
        node_env = tmp_path / ".env"
        node_env.write_text("IMMICH_URL=https://from-node.test\nIMMICH_API_KEY=node-key\n")
        with patch.object(node_mod, "_config_paths", return_value=(None, str(node_env))):
            config = node_mod.resolve_config()
        assert config["url_source"] == config["key_source"] == "dotenv"

    def test_environment_alone_configures_nothing(self, tmp_path):
        with (
            patch.object(node_mod, "_config_paths", return_value=(None, str(tmp_path / ".env"))),
            patch.dict(
                "os.environ", {"IMMICH_URL": "https://from-env.test", "IMMICH_API_KEY": "k"}
            ),
        ):
            config = node_mod.resolve_config()
        assert (config["url"], config["key"]) == ("", "")
        assert config["url_source"] == config["key_source"] == "none"

    def test_routes_register_is_a_noop_outside_comfyui(self):
        from immich_nodes import routes

        assert routes.register() is False
