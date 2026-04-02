# comfyui-immich

ComfyUI custom node that uploads generated images directly to an Immich server.

## Install

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/your-org/comfyui-immich.git
```

No additional Python dependencies required -- uses only stdlib plus packages already in ComfyUI (PIL, numpy, torch).

## Configure

```bash
cd custom_nodes/comfyui-immich
cp .env.example .env
```

Edit `.env` with your Immich instance URL and API key:

```
IMMICH_URL=https://immich.example.com
IMMICH_API_KEY=your-api-key-here
```

## Node: Save to Immich

**Category:** `image/immich`

Uploads images to Immich at the end of a workflow (output/terminal node).

### Inputs

| Input | Type | Required | Description |
|-------|------|----------|-------------|
| images | IMAGE | Yes | Image tensor from the pipeline |
| description | STRING | No | Metadata description to set on the uploaded asset |
| album_id | STRING | No | Immich album UUID to add the image to |
| filename_prefix | STRING | No | Prefix for the uploaded filename (default: "ComfyUI") |

### Behavior

- Converts each image in the batch from tensor to PNG and uploads via the Immich API
- Optionally sets a description and/or adds to an album
- Logs upload status to the ComfyUI terminal
- Handles errors per-image without crashing the workflow
