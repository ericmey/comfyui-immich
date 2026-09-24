"""The real aiohttp adapter, served over HTTP. Skipped where aiohttp is absent (e.g. CI)."""

import asyncio
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
