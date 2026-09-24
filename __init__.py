try:
    from .immich_nodes.routes import register as _register_routes
    from .immich_nodes.save_to_immich import SaveToImmich
except ImportError:
    from immich_nodes.routes import register as _register_routes
    from immich_nodes.save_to_immich import SaveToImmich

# Read-only status routes (GET /immich/status, POST /immich/test); no-op outside ComfyUI.
_register_routes()

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "SaveToImmich": SaveToImmich,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveToImmich": "Save to Immich",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
