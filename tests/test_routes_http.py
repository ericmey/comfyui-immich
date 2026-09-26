"""The real aiohttp adapter, served over HTTP. Skipped where aiohttp is absent (e.g. CI)."""

import asyncio
import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

web = pytest.importorskip("aiohttp.web")

from immich_nodes import save_to_immich as node_mod  # noqa: E402
from immich_nodes import status  # noqa: E402

SENTINEL = "SENTINEL-route-KEY-51vq"


def _serve_and_call(calls):
    """Register routes on a fake PromptServer, serve them, run `calls(base_url)`."""
    routes = web.RouteTableDef()
    fake_server = types.ModuleType("server")
    fake_server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes))

    async def main():
        with patch.dict(sys.modules, {"server": fake_server}):
            from immich_nodes import routes as routes_mod

            assert routes_mod.register() is True
        app = web.Application()
        app.add_routes(routes)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            return await calls(f"http://127.0.0.1:{port}")
        finally:
            await runner.cleanup()

    return asyncio.run(main())


@pytest.fixture(autouse=True)
def _configured(tmp_path):
    status._last_test[0] = None
    env = {"IMMICH_URL": "https://immich.example.test", "IMMICH_API_KEY": SENTINEL}
    with (
        patch.dict("immich_nodes.save_to_immich.os.environ", env),
        patch.object(node_mod, "_config_paths", return_value=(None, str(tmp_path / ".env"))),
    ):
        yield


def test_status_and_test_routes_over_http():
    import aiohttp

    ok = MagicMock(status=200)
    ok.__enter__ = lambda s: s
    ok.__exit__ = MagicMock(return_value=False)

    async def calls(base):
        out = {}
        async with aiohttp.ClientSession() as s:
            async with s.get(base + "/immich/status") as r:
                out["status"] = (r.status, await r.text())
            async with s.post(base + "/immich/test?url=http://evil.test") as r:
                out["query"] = r.status
            async with s.post(base + "/immich/test", data=b'{"url":"http://evil.test"}') as r:
                out["body"] = r.status
            async with s.post(base + "/immich/test") as r:
                out["plain"] = (r.status, await r.json())
        return out

    with patch("immich_nodes.save_to_immich.urlopen", return_value=ok) as mock_open:
        out = _serve_and_call(calls)

    assert out["status"][0] == 200
    assert SENTINEL not in out["status"][1] and SENTINEL[-4:] not in out["status"][1]
    assert out["query"] == 400
    assert out["body"] == 400
    assert out["plain"] == (200, {"ok": True, "status": 200, "error": None})
    assert mock_open.call_count == 1  # only the input-free call reached the network
    assert mock_open.call_args.args[0].full_url == "https://immich.example.test/api/users/me"


def _raw_post(base, head, body_part):
    """Send raw request bytes and return the first response line, or a timeout marker."""
    import socket as _socket

    host, port = base.removeprefix("http://").split(":")
    with _socket.create_connection((host, int(port)), timeout=3) as sock:
        sock.sendall(head + body_part)
        try:
            return sock.recv(64).split(b"\r\n")[0]
        except TimeoutError:
            return b"TIMEOUT (server waited for the body)"


def test_declared_body_is_refused_without_reading_it():
    head = (
        b"POST /immich/test HTTP/1.1\r\nHost: x\r\n"
        b"Content-Type: application/octet-stream\r\nContent-Length: 100000000\r\n\r\n"
    )

    async def calls(base):
        return await asyncio.to_thread(_raw_post, base, head, b"only-a-few-bytes")

    with patch("immich_nodes.save_to_immich.urlopen") as mock_open:
        first_line = _serve_and_call(calls)
    assert first_line.startswith(b"HTTP/1.1 400"), first_line
    mock_open.assert_not_called()


def test_chunked_body_is_refused_after_one_byte():
    head = b"POST /immich/test HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"

    async def calls(base):
        # One chunk, and the terminating zero-chunk is never sent.
        return await asyncio.to_thread(_raw_post, base, head, b"5\r\nhello\r\n")

    with patch("immich_nodes.save_to_immich.urlopen") as mock_open:
        first_line = _serve_and_call(calls)
    assert first_line.startswith(b"HTTP/1.1 400"), first_line
    mock_open.assert_not_called()


def test_settings_route_saves_same_origin_json_only(tmp_path):
    import aiohttp

    user_env = tmp_path / "user" / "comfyui-immich.env"
    body = {"url": "https://new.example", "confirm_url_change": True, "api_key": SENTINEL}

    async def calls(base):
        out = {}
        async with aiohttp.ClientSession() as s:
            async with s.post(
                base + "/immich/settings", json=body, headers={"Origin": "https://evil.example"}
            ) as r:
                out["cross"] = (r.status, await r.json())
            async with s.post(
                base + "/immich/settings", data=json.dumps(body), headers={"Origin": base}
            ) as r:
                out["form"] = r.status  # no JSON content type
            async with s.post(base + "/immich/settings", json=body, headers={"Origin": base}) as r:
                out["ok"] = (r.status, await r.text())
        return out

    with (
        patch.dict("immich_nodes.save_to_immich.os.environ", {}, clear=True),
        patch.object(
            node_mod, "_config_paths", return_value=(str(user_env), str(tmp_path / ".env"))
        ),
    ):
        out = _serve_and_call(calls)
        config = node_mod.resolve_config()

    assert out["cross"] == (403, {"ok": False, "error": "cross_origin"})
    assert out["form"] == 415
    assert out["ok"][0] == 200 and SENTINEL not in out["ok"][1]
    assert config["url"] == "https://new.example" and config["key"] == SENTINEL
