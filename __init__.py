from .nodes.save_to_immich import SaveToImmich

NODE_CLASS_MAPPINGS = {
    "SaveToImmich": SaveToImmich,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveToImmich": "Save to Immich",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
