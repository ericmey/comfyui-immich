"""Editable Immich settings for ComfyUI's Settings dialog.

The panel saves IMMICH_URL and IMMICH_API_KEY to ComfyUI's user directory
(`user/comfyui-immich.env`), which survives reinstalling the node. ComfyUI has
no login by default, so the save path is written against the obvious abuses:

- it accepts only same-origin JSON from a browser: the Origin's scheme, host
  and port must match this server, or an origin listed in
  IMMICH_ALLOWED_ORIGINS (for a TLS reverse proxy). Cross-site pages and
  requests without an Origin are refused. X-Forwarded-* headers are ignored,
  because any client can send them;
- the API key is write-only: it is never returned, logged or shown;
- changing the URL needs explicit confirmation AND clears the saved key unless
  a new key is sent with it, so redirecting the URL cannot forward your
  existing key to someone else's server. If the key is still in the node
  folder's legacy .env, which the panel cannot remove, the URL cannot be
  changed and the key cannot be cleared from the panel;
- the status never reports absolute paths.

This is only as safe as your ComfyUI exposure: anyone who can use your ComfyUI
page can also change these settings.
"""

import json
import os
from urllib.parse import urlsplit

from . import save_to_immich as _node

MAX_BODY_BYTES = 16 * 1024
WRITABLE_KEYS = ("IMMICH_URL", "IMMICH_API_KEY")
_FIELDS = {"url", "api_key", "clear_api_key", "confirm_url_change"}
_DEFAULT_PORTS = {"http": 80, "https": 443}


class SettingsWriteError(RuntimeError):
    """The settings could not be saved; `code` is a short machine-readable reason."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def valid_url(value):
    """An Immich URL the panel may save: absolute http(s), no credentials, query or fragment."""
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    return (
        parts.scheme in ("http", "https")
        and bool(parts.hostname)
        and parts.username is None
        and parts.password is None
        and not parts.query
        and not parts.fragment
    )


def key_outside_panel():
    """Name the source of an API key the panel cannot remove, or None.

    A key in the node folder's legacy .env stays in force after the panel
    deletes its own copy, so it would follow a new URL.
    """
    _, node_env = _node._config_paths()
    if _node._load_env(node_env).get("IMMICH_API_KEY", "").strip():
        return "dotenv"
    return None


def config_location():
    """Where settings are saved, without the absolute path (which names the machine's user)."""
    user_env, _ = _node._config_paths()
    if user_env is not None:
        return f"ComfyUI user directory ({_node._USER_CONFIG_NAME})"
    return "node folder (.env)"


def allowed_origins():
    """Extra browser origins allowed to save settings (TLS reverse proxy).

    Set by hand as ``IMMICH_ALLOWED_ORIGINS=https://a.example`` in
    ``user/comfyui-immich.env``. Deliberately not writable from the panel.
    """
    user_env, node_env = _node._config_paths()
    for values in (
        _node._load_env(user_env) if user_env else {},
        _node._load_env(node_env),
    ):
        raw = (values.get("IMMICH_ALLOWED_ORIGINS") or "").strip()
        if raw:
            return [o.strip() for o in raw.split(",") if o.strip()]
    return []


def write_user_settings(updates):
    """Apply `updates` to the user-directory settings file; None or "" removes a key.

    Written atomically and readable only by its owner, because it may hold the API key.
    """
    user_env, _ = _node._config_paths()
    if user_env is None:
        raise SettingsWriteError("no_user_directory", "ComfyUI's user directory is not available")
    for key, value in updates.items():
        if key not in WRITABLE_KEYS:
            raise SettingsWriteError("unknown_setting", f"{key} is not a writable setting")
        if value is not None and ("\n" in value or "\r" in value):
            raise SettingsWriteError("invalid_value", f"{key} must be a single line")

    current = _node._load_env(user_env)
    for key, value in updates.items():
        if value:
            current[key] = value
        else:
            current.pop(key, None)

    os.makedirs(os.path.dirname(user_env), exist_ok=True)
    body = "# Written by the Save to Immich settings panel (Settings -> Immich).\n"
    body += "".join(f"{key}={value}\n" for key, value in current.items())
    tmp = os.path.join(os.path.dirname(user_env), f".{os.path.basename(user_env)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(body)
    os.chmod(tmp, 0o600)
    os.replace(tmp, user_env)


def _origin(scheme, netloc):
    """Normalise to (scheme, host, port) as a browser compares origins; None if malformed."""
    scheme = (scheme or "").lower()
    if scheme not in _DEFAULT_PORTS or not netloc:
        return None
    try:
        parts = urlsplit(f"//{netloc}")
        port = parts.port
    except ValueError:
        return None
    if not parts.hostname or parts.username is not None or parts.password is not None:
        return None
    return scheme, parts.hostname, port or _DEFAULT_PORTS[scheme]


def same_origin(origin_header, scheme, host):
    """A browser always sends Origin on a cross-site POST; require it to match this server."""
    try:
        sent = urlsplit(origin_header or "")
    except ValueError:
        return False
    if sent.path or sent.query or sent.fragment:
        return False
    got = _origin(sent.scheme, sent.netloc)
    if got is None:
        return False
    if got == _origin(scheme, host or ""):
        return True
    for allowed in allowed_origins():
        try:
            parts = urlsplit(allowed)
        except ValueError:
            continue
        if got == _origin(parts.scheme, parts.netloc):
            return True
    return False


def plan_settings_update(body):
    """Turn a panel request into file updates, or return an error code.

    Pure apart from reading the current configuration, so the rules are unit-testable.
    """
    if set(body) - _FIELDS:
        return {}, "unknown_field"
    updates = {}

    new_key = body.get("api_key")
    if new_key is not None and (not isinstance(new_key, str) or not new_key.strip()):
        return {}, "invalid_api_key"
    if body.get("clear_api_key") is True:
        # Clearing only the panel's copy would report success while the key stays in force.
        if key_outside_panel():
            return {}, "key_outside_panel"
        updates["IMMICH_API_KEY"] = None

    if "url" in body:
        url = body["url"]
        if not isinstance(url, str):
            return {}, "invalid_url"
        url = _node._normalize_immich_url(url)
        if url and not valid_url(url):
            return {}, "invalid_url"
        if url != _node.resolve_config()["url"]:
            if body.get("confirm_url_change") is not True:
                return {}, "confirm_url_change"
            # Never let a new server receive the key that was saved for the old one.
            # A key in the legacy node .env would follow the URL once the panel's
            # copy is deleted, so refuse until it is moved or removed.
            if key_outside_panel():
                return {}, "key_outside_panel"
            updates["IMMICH_API_KEY"] = None
        updates["IMMICH_URL"] = url or None

    if new_key is not None:
        updates["IMMICH_API_KEY"] = new_key.strip()
    return updates, None


def save_request(origin, scheme, host, content_type, raw):
    """Return (http_status, payload) for POST /immich/settings.

    `raw` is the body, read up to MAX_BODY_BYTES + 1 bytes.
    """
    from .status import status_payload

    def refuse(code, error):
        return code, {"ok": False, "error": error}

    if not same_origin(origin, scheme, host):
        return refuse(403, "cross_origin")
    if content_type != "application/json":
        return refuse(415, "json_required")
    if len(raw) > MAX_BODY_BYTES:
        return refuse(413, "too_large")
    try:
        body = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return refuse(400, "invalid_json")
    if not isinstance(body, dict):
        return refuse(400, "invalid_json")
    updates, error = plan_settings_update(body)
    if error:
        conflict = ("confirm_url_change", "key_outside_panel")
        return refuse(409 if error in conflict else 400, error)
    try:
        write_user_settings(updates)
    except SettingsWriteError as exc:
        return refuse(503 if exc.code == "no_user_directory" else 400, exc.code)
    return 200, {"ok": True, **status_payload()}
