"""Retry archiving a saved PNG without running diffusion again."""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from .save_to_immich import SaveToImmich


def _read_graph(path):
    """Return the ComfyUI graph embedded in a saved PNG.

    Recovery is only meaningful against the original file, so every way the
    graph can be missing or malformed is an explicit error. Falling back to an
    empty graph would archive the asset with no metadata and report success --
    the exact outcome this tool exists to repair.
    """
    with Image.open(path) as image:
        if image.format != "PNG":
            raise ValueError("archive recovery requires the original PNG")
        embedded = image.info.get("prompt")

    if not embedded:
        raise ValueError("no ComfyUI graph embedded in this PNG: nothing to recover")
    try:
        graph = json.loads(embedded)
    except json.JSONDecodeError as exc:
        raise ValueError(f"embedded ComfyUI graph is not valid JSON: {exc}") from exc
    if not isinstance(graph, dict):
        raise ValueError("embedded ComfyUI graph is not a mapping of nodes")
    return graph


def retry(path, *, asset_id=None):
    path = Path(path)
    graph = _read_graph(path)
    outputs = [
        node["inputs"]
        for node in graph.values()
        if isinstance(node, dict)
        and node.get("class_type") == "SaveToImmich"
        and isinstance(node.get("inputs"), dict)
    ]
    if len(outputs) > 1:
        raise ValueError("multiple archive nodes: recover through the original workflow")
    inputs = outputs[0] if outputs else {}
    node = SaveToImmich()
    description = inputs.get("description") or node._build_auto_description(
        graph, character=inputs.get("character", "")
    )
    album_id = inputs.get("album_id", "")
    if asset_id and not description and not album_id:
        raise ValueError(
            "nothing to retry: --asset-id skips the upload and the graph "
            "carries no description or album"
        )
    return node.archive_png(
        path.read_bytes(),
        path.name,
        description=description,
        album_id=album_id,
        asset_id=asset_id,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("png", help="Original saved PNG, including its embedded graph")
    parser.add_argument("--asset-id", help="Existing asset ID; retry metadata without uploading")
    args = parser.parse_args()

    # OSError covers an unreadable or non-image file; ValueError is every
    # recoverable-input complaint above. Neither deserves a traceback.
    try:
        report = retry(args.png, asset_id=args.asset_id)
    except (OSError, ValueError) as exc:
        print(f"retry_archive: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report, indent=2))
    if report["description"] == "not_requested":
        print(
            "retry_archive: the embedded graph carries no description; "
            "nothing was written to the asset",
            file=sys.stderr,
        )
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
