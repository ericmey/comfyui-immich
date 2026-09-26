"""Saving Immich settings from ComfyUI's Settings panel (immich_nodes/settings.py)."""

import json
import os
from unittest.mock import patch

import pytest

from immich_nodes import save_to_immich as node_mod
from immich_nodes import settings, status

SENTINEL = "SENTINEL-settings-KEY-7c2e"
ORIGIN = "http://127.0.0.1:8188"
HOST = "127.0.0.1:8188"


@pytest.fixture
def paths(tmp_path):
    """A writable ComfyUI user directory, an empty node folder, and a clean environment."""
    user_env = tmp_path / "user" / "comfyui-immich.env"
    node_env = tmp_path / "node" / ".env"
    node_env.parent.mkdir()
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(node_mod, "_config_paths", return_value=(str(user_env), str(node_env))),
    ):
        yield user_env, node_env


def save(body, origin=ORIGIN, scheme="http", host=HOST, content_type="application/json"):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return settings.save_request(origin, scheme, host, content_type, raw)


def saved(user_env):
    return node_mod._load_env(str(user_env))


def test_save_writes_user_dir_and_never_returns_the_key(paths):
    user_env, _ = paths
    code, body = save(
        {"url": "https://immich.example/", "confirm_url_change": True, "api_key": SENTINEL}
    )
    assert code == 200 and body["ok"] is True
    assert body["url"] == "https://immich.example" and body["key_set"] is True
    assert SENTINEL not in json.dumps(body)
    assert SENTINEL not in json.dumps(status.status_payload())
    assert saved(user_env) == {"IMMICH_URL": "https://immich.example", "IMMICH_API_KEY": SENTINEL}
    assert (user_env.stat().st_mode & 0o777) == 0o600  # the file holds the key


def test_trailing_api_is_normalised_like_the_node_does(paths):
    user_env, _ = paths
    save({"url": "https://immich.example/api", "confirm_url_change": True, "api_key": SENTINEL})
    assert saved(user_env)["IMMICH_URL"] == "https://immich.example"


@pytest.mark.parametrize(
    ("kw", "code", "error"),
    [
        ({"origin": ""}, 403, "cross_origin"),
        ({"origin": "https://evil.example"}, 403, "cross_origin"),
        # Same host and port, different scheme: a different origin to a browser.
        ({"origin": "https://same.example:8188", "host": "same.example:8188"}, 403, "cross_origin"),
        ({"origin": "http://127.0.0.1:8188/x"}, 403, "cross_origin"),
        ({"origin": "null"}, 403, "cross_origin"),
        ({"content_type": "text/plain"}, 415, "json_required"),
    ],
    ids=[
        "no origin",
        "cross-site",
        "scheme differs",
        "origin with path",
        "opaque origin",
        "form post",
    ],
)
def test_requests_a_malicious_page_could_send_are_refused(paths, kw, code, error):
    user_env, _ = paths
    got_code, body = save({"api_key": SENTINEL}, **kw)
    assert (got_code, body["error"]) == (code, error)
    assert not user_env.exists()


@pytest.mark.parametrize(
    ("kw", "env"),
    [
        ({"origin": "http://same.example", "host": "same.example:80"}, {}),
        ({"origin": "https://same.example", "host": "same.example:443", "scheme": "https"}, {}),
        ({"origin": "http://[::1]:8188", "host": "[::1]:8188"}, {}),
        # TLS reverse proxy: the browser sees https, ComfyUI sees plain http.
        (
            {"origin": "https://comfy.example", "host": "comfy.example"},
            {"IMMICH_ALLOWED_ORIGINS": "https://comfy.example"},  # set by hand in the user file
        ),
    ],
    ids=["default http port", "default https port", "ipv6", "allowed proxy origin"],
)
def test_same_origin_accepts_the_page_itself(paths, kw, env):
    user_env, _ = paths
    if env:
        user_env.parent.mkdir(parents=True, exist_ok=True)
        user_env.write_text("".join(f"{k}={v}\n" for k, v in env.items()))
    code, _ = save({"api_key": SENTINEL}, **kw)
    assert code == 200


def test_oversized_and_malformed_bodies_are_refused(paths):
    assert save(b"{" + b" " * settings.MAX_BODY_BYTES + b"}")[0] == 413
    assert save(b"not json")[1]["error"] == "invalid_json"
    assert save(b"[1]")[1]["error"] == "invalid_json"


def test_changing_the_url_needs_confirmation(paths):
    user_env, _ = paths
    code, body = save({"url": "https://immich.example"})
    assert (code, body["error"]) == (409, "confirm_url_change")
    assert not user_env.exists()


def test_changing_the_url_drops_the_old_key_unless_a_new_one_is_sent(paths):
    user_env, _ = paths
    save({"url": "https://good.example", "confirm_url_change": True, "api_key": SENTINEL})
    code, body = save({"url": "https://attacker.example", "confirm_url_change": True})
    assert code == 200 and body["key_set"] is False
    assert "IMMICH_API_KEY" not in saved(user_env)
    save({"url": "https://good.example", "confirm_url_change": True, "api_key": "new-key"})
    assert saved(user_env)["IMMICH_API_KEY"] == "new-key"


def test_saving_the_same_url_keeps_the_key(paths):
    save({"url": "https://good.example", "confirm_url_change": True, "api_key": SENTINEL})
    code, body = save(
        {"url": "https://good.example/"}
    )  # unchanged after normalising: no confirm needed
    assert code == 200 and body["key_set"] is True


def test_clear_key(paths):
    save({"url": "https://good.example", "confirm_url_change": True, "api_key": SENTINEL})
    code, body = save({"clear_api_key": True})
    assert code == 200 and body["key_set"] is False


@pytest.mark.parametrize("new_key", [None, "replacement"], ids=["no new key", "with new key"])
def test_url_change_refused_when_the_key_lives_in_the_legacy_node_dotenv(paths, new_key):
    # Deleting the panel's copy would leave the node .env key in force, and it
    # would be sent to the new server.
    _, node_env = paths
    settings.write_user_settings({"IMMICH_URL": "https://good.example"})
    node_env.write_text("IMMICH_API_KEY=OLD-KEY\n")
    body = {"url": "https://new.example", "confirm_url_change": True}
    if new_key:
        body["api_key"] = new_key
    assert settings.plan_settings_update(body) == ({}, "key_outside_panel")
    code, got = save(body)
    assert (code, got["error"]) == (409, "key_outside_panel")
    assert node_mod.resolve_config()["url"] == "https://good.example"


def test_an_environment_key_neither_blocks_nor_follows_a_url_change(paths):
    settings.write_user_settings(
        {"IMMICH_URL": "https://good.example", "IMMICH_API_KEY": "PANEL-KEY"}
    )
    with patch.dict(os.environ, {"IMMICH_API_KEY": "ENV-KEY", "IMMICH_URL": "https://env.example"}):
        code, got = save({"url": "https://new.example", "confirm_url_change": True})
        assert code == 200 and got["url"] == "https://new.example" and got["key_set"] is False
        assert node_mod.resolve_config()["key"] == ""


def test_clear_key_refused_when_the_key_lives_in_the_legacy_node_dotenv(paths):
    user_env, node_env = paths
    settings.write_user_settings({"IMMICH_API_KEY": "PANEL-KEY"})
    node_env.write_text("IMMICH_API_KEY=NODE-KEY\n")
    code, got = save({"clear_api_key": True})
    assert (code, got["error"]) == (409, "key_outside_panel")
    assert saved(user_env)["IMMICH_API_KEY"] == "PANEL-KEY"


def test_clear_key_ignores_an_environment_key(paths):
    settings.write_user_settings({"IMMICH_API_KEY": "PANEL-KEY"})
    with patch.dict(os.environ, {"IMMICH_API_KEY": "ENV-KEY"}):
        code, got = save({"clear_api_key": True})
    assert code == 200 and got["key_set"] is False


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ({"url": "https://user:pw@immich.example", "confirm_url_change": True}, "invalid_url"),
        ({"url": "https://immich.example/?token=x", "confirm_url_change": True}, "invalid_url"),
        ({"url": "ftp://immich.example", "confirm_url_change": True}, "invalid_url"),
        ({"url": 5, "confirm_url_change": True}, "invalid_url"),
        ({"api_key": "   "}, "invalid_api_key"),
        ({"api_key": "a\nIMMICH_URL=https://evil.example"}, "invalid_value"),
        ({"key": SENTINEL}, "unknown_field"),
    ],
)
def test_invalid_input_is_refused_and_nothing_is_written(paths, body, error):
    user_env, _ = paths
    code, got = save(body)
    assert code == 400 and got["error"] == error
    assert not user_env.exists()


def test_environment_variables_no_longer_override_the_panel(paths):
    save({"url": "https://panel.example", "confirm_url_change": True, "api_key": SENTINEL})
    with patch.dict(os.environ, {"IMMICH_URL": "https://env.example", "IMMICH_API_KEY": "env"}):
        payload = status.status_payload()
    assert payload["url"] == "https://panel.example"
    assert payload["source"] == {"url": "userdir", "key": "userdir"}
    assert "shadowed" not in payload


def test_status_never_reports_an_absolute_path(paths, tmp_path):
    _, body = save(
        {"url": "https://immich.example", "confirm_url_change": True, "api_key": SENTINEL}
    )
    for payload in (body, status.status_payload()):
        text = json.dumps(payload)
        assert str(tmp_path) not in text
        assert "config_path" not in payload and "user_config_path" not in payload
        assert payload["config_location"] == "ComfyUI user directory (comfyui-immich.env)"


def test_no_user_directory_means_no_save(tmp_path):
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(node_mod, "_config_paths", return_value=(None, str(tmp_path / ".env"))),
    ):
        code, body = save({"api_key": SENTINEL})
        assert status.status_payload()["writable"] is False
    assert (code, body["error"]) == (503, "no_user_directory")


def test_url_edit_applies_even_when_the_environment_sets_a_url(paths):
    """Before 0.6.0 this was refused as url_shadowed; the environment is no longer read."""
    user_env, _ = paths
    settings.write_user_settings({"IMMICH_API_KEY": "PANEL-KEY"})
    with patch.dict(os.environ, {"IMMICH_URL": "https://env.example"}):
        code, got = save({"url": "https://panel.example", "confirm_url_change": True})
    assert code == 200 and got["url"] == "https://panel.example"
    assert saved(user_env)["IMMICH_URL"] == "https://panel.example"
