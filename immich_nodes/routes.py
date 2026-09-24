"""Register the read-only status routes with ComfyUI's server, when present."""

import asyncio

from .status import run_connection_test, status_payload


def register():
    """Add GET /immich/status and POST /immich/test. Returns False outside ComfyUI."""
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
        body = await request.read()
        has_input = bool(request.query_string) or bool(body)
        code, payload = await asyncio.to_thread(run_connection_test, has_input)
        return web.json_response(payload, status=code)

    return True
