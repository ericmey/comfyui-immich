# comfyui-immich

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that integrate with [Immich](https://immich.app) — the self-hosted photo management platform.

## Features

- **Save to Immich** — Upload generated images directly to your Immich server
- Full workflow and prompt metadata embedded in PNG (drag-drop back into ComfyUI to reproduce)
- Optional album assignment and description tagging on upload
- Per-image error handling — one failure doesn't crash the batch
- Zero extra dependencies — uses only packages already in ComfyUI (PIL, numpy, torch)

## Installation

Clone into your ComfyUI `custom_nodes` directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/ericmey/comfyui-immich.git
```

Restart ComfyUI. The node will appear under **image/immich** in the node menu.

## Configuration

Create a `.env` file in the node directory with your Immich credentials:

```bash
cd custom_nodes/comfyui-immich
cp .env.example .env
```

Edit `.env`:

```env
IMMICH_URL=https://your-immich-instance.com
IMMICH_API_KEY=your-api-key-here
```

The `.env` file is gitignored and persists across `git pull` updates.

### Getting an Immich API Key

1. Open your Immich instance in a browser
2. Go to **User Settings** (click your avatar → Account Settings)
3. Scroll to **API Keys** → **New API Key**
4. Give it a name (e.g., "ComfyUI") and create
5. Copy the key into your `.env` file

## Nodes

### Save to Immich

**Category:** `image/immich`

An output node that uploads images to Immich at the end of a workflow.

#### Inputs

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| images | IMAGE | Yes | — | Image tensor from the pipeline |
| description | STRING | No | `""` | Description visible in Immich UI |
| album_id | STRING | No | `""` | Immich album UUID to add the image to |
| filename_prefix | STRING | No | `"ComfyUI"` | Prefix for the uploaded filename |

#### Hidden Inputs (automatic)

| Input | Description |
|-------|-------------|
| prompt | Full node/prompt data — embedded in PNG metadata |
| extra_pnginfo | Workflow JSON — embedded in PNG metadata |

#### What Gets Saved

Each uploaded image includes:

- **PNG metadata**: Full ComfyUI workflow + prompt data (same format as the built-in SaveImage node). You can drag the image back into ComfyUI to load the exact workflow that created it.
- **Immich description**: Whatever you put in the `description` field, visible in the Immich web UI.
- **Album placement**: If `album_id` is provided, the image is added to that album immediately after upload.

#### Usage

Wire it as a terminal node — connect the image output from your VAE Decode, Detailer, or any image-producing node:

```
KSampler → VAE Decode → Save to Immich
```

It can be used alongside or instead of the built-in SaveImage node.

## Updating

```bash
cd /path/to/ComfyUI/custom_nodes/comfyui-immich
git pull
```

Your `.env` file is preserved — it's in `.gitignore`.

## License

MIT
