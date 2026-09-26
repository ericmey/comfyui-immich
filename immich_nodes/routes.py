"""Register the status, test and settings routes with ComfyUI's server, when present."""

import asyncio

from .settings import MAX_BODY_BYTES, save_request
from .status import run_connection_test, status_payload


def register():
    """Add GET /immich/status, POST /immich/test and POST /immich/settings.

    Returns False outside ComfyUI.
    """
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return False

    routes = PromptServer.instance.routes

    @routes.get("/immich/status")
    async def _status(_request):
        return web.json_response(status_payload())

    @routes.post("/immich/test")
    async def _test(request):
        # Never buffer a caller's body: a declared length is refused unread, and
        # a chunked or unknown-length body is refused after at most one byte.
        has_input = bool(request.query_string) or (request.content_length or 0) > 0
        if not has_input and request.content_length is None:
            has_input = bool(await request.content.read(1))
        code, payload = await asyncio.to_thread(run_connection_test, has_input)
        return web.json_response(payload, status=code)

    @routes.post("/immich/settings")
    async def _settings(request):
        # read(n) returns whatever is buffered, which can be part of the body;
        # keep reading until EOF or one byte past the limit.
        raw = b""
        while len(raw) <= MAX_BODY_BYTES:
            chunk = await request.content.read(MAX_BODY_BYTES + 1 - len(raw))
            if not chunk:
                break
            raw += chunk
        code, payload = save_request(
            request.headers.get("Origin", ""),
            request.scheme,
            request.host,
            request.content_type,
            raw,
        )
        return web.json_response(payload, status=code)

    return True
