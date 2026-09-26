"""Connection status and test for the settings panel.

Two operations, both deliberately incapable of changing configuration (saving
lives in settings.py, behind its own same-origin and confirmation rules):

* status_payload(): what is configured and where it came from. Never returns
  the API key or anything derived from it.
* run_connection_test(has_input): one authenticated GET to the *saved* Immich
  URL. It takes no URL or key from the caller (any input is refused), is
  rate-limited, and reports only a status code and an error class, never the
  remote body.

ComfyUI has no authentication by default, so anything reachable here is
reachable by anyone who can reach ComfyUI. The status never reports absolute
paths, which would name the machine's user.
"""

import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request

from . import save_to_immich as _node

TEST_WINDOW_SECONDS = 5.0
_TEST_PATH = "/api/users/me"  # 200 with a valid key, 401 without
_last_test: list = [None]
_test_lock = threading.Lock()


def status_payload():
    from . import settings

    config = _node.resolve_config()
    return {
        "url": config["url"] or None,
        "key_set": bool(config["key"]),
        "source": {"url": config["url_source"], "key": config["key_source"]},
        "writable": config["user_env"] is not None,
        "config_location": settings.config_location(),
    }


def _result(ok, status=None, error=None):
    return {"ok": ok, "status": status, "error": error}


def _classify_http(code):
    if code == 401:
        return "unauthorized"
    if code == 403:
        return "forbidden"
    if 300 <= code < 400:
        return "redirect_refused"
    return "bad_response"


def run_connection_test(has_input, clock=time.monotonic):
    """Return (http_status, payload) for POST /immich/test."""
    if has_input:
        return 400, _result(False, error="input_not_accepted")
    with _test_lock:
        now = clock()
        last = _last_test[0]
        if last is not None and now - last < TEST_WINDOW_SECONDS:
            return 429, _result(False, error="rate_limited")
        _last_test[0] = now

    config = _node.resolve_config()
    if not config["url"] or not config["key"]:
        return 200, _result(False, error="not_configured")

    req = Request(config["url"] + _TEST_PATH, headers={"x-api-key": config["key"]}, method="GET")
    try:
        with _node.urlopen(req, timeout=_node._REQUEST_TIMEOUT_SECONDS) as resp:
            return 200, _result(True, status=resp.status)
    except HTTPError as exc:
        exc.close()
        return 200, _result(False, status=exc.code, error=_classify_http(exc.code))
    except TimeoutError:
        return 200, _result(False, error="timeout")
    except URLError as exc:
        timed_out = isinstance(exc.reason, TimeoutError)
        return 200, _result(False, error="timeout" if timed_out else "unreachable")
    except OSError:
        return 200, _result(False, error="unreachable")
