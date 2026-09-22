# comfyui-immich

Custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that integrate with [Immich](https://immich.app) — the self-hosted photo management platform.

## Features

- **Save to Immich** — Upload generated images directly to your Immich server
- Full workflow and prompt metadata embedded in PNG (drag-drop back into ComfyUI to reproduce)
- ComfyUI-viewable local preview written before upload, so a down Immich
  does not eat the render (Atelier and the UI both read this file)
- Optional character label, album assignment, and description tagging
- Immich v3-compatible upload payloads
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
| character | STRING | No | `""` | Who this render is of. Prefixed onto the Immich description as `Character: …`. Atelier fills this from the record; type it by hand in the UI. Not an Immich people/face tag. |
| description | STRING | No | `""` | Description visible in Immich UI. Empty means auto-build from the graph (character, checkpoint/UNET, sampler, seed, prompt). |
| album_id | STRING | No | `""` | Immich album UUID to add the image to |
| filename_prefix | STRING | No | `"ComfyUI"` | Prefix for the uploaded filename. May include subfolders (`portraits/nova`) within ComfyUI's output directory; absolute paths, `..`, and escaping symlinks are refused. |

#### Hidden Inputs (automatic)

| Input | Description |
|-------|-------------|
| prompt | Full node/prompt data — embedded in PNG metadata |
| extra_pnginfo | Workflow JSON — embedded in PNG metadata |

#### What Gets Saved

Each uploaded image includes:

- **PNG metadata**: Full ComfyUI workflow + prompt data (same format as the built-in SaveImage node). You can drag the image back into ComfyUI to load the exact workflow that created it.
- **ComfyUI preview**: A local output copy written *before* the upload. The UI and Atelier download this file from history even if Immich is unreachable.
- **Immich description**: The `description` field if you filled it, otherwise an auto-built caption from the graph. A filled `character` is always prefixed.
- **Album placement**: If `album_id` is provided, the image is added to that album immediately after upload.

#### Usage

Wire it as a terminal node — connect the image output from your VAE Decode, Detailer, or any image-producing node:

```
KSampler → VAE Decode → Save to Immich
```

Use it **instead of** SaveImage when you want one copy. Running both publishes
the same picture twice (re-encoded) because each output node writes its own
history entry — Atelier and a hand-run graph both see that as two files.

Hand-run graphs and Atelier share this node. Atelier only fills `character`
(and optional `description` / `album_id`). A graph opened in the ComfyUI UI
with those fields left blank still archives: the node reads the prompt,
checkpoint or UNET, sampler, and seed itself.

## Archive receipts and recovery

Local preview delivery and Immich archiving have separate outcomes. History
contains `ui.images` for previews and `ui.archive` for per-image upload,
description, and album results. Each archive receipt includes an asset ID
when available and stage-specific errors. A preview alone does not confirm
archiving. Missing archive configuration also preserves the local preview.

The two directions are independent: a rejected `filename_prefix` or an
unwritable output directory is reported as a `preview` stage error while the
image still reaches Immich. One failed image never cancels the rest of the
batch, so receipts for images already uploaded are never lost.

The node prints one summary line per batch, so a working archive is never
silent. Failures name the file — that name is the argument `retry_archive`
takes — and a retry hint appears whenever the local PNG still exists.

Receipt values worth knowing:

| Field | Value | Meaning |
|-------|-------|---------|
| `upload` | `ok` / `reused` / `failed` | `reused` means an existing asset ID was supplied and nothing was uploaded |
| `description` | `ok` / `failed` / `not_requested` / `not_attempted` | `not_attempted` means an earlier stage failed first |
| `album` | `ok` / `unconfirmed` / `failed` / `not_requested` / `not_attempted` | `unconfirmed` means Immich returned success with no per-asset body to verify |

Filename prefixes may include subfolders within ComfyUI's output directory.
Absolute paths, parent components, and symlinks escaping that directory are
refused. History reports the correct basename and subfolder for retrieval.

Retry an existing PNG without another GPU render from this node's directory:

```bash
/path/to/ComfyUI/.venv/bin/python -m immich_nodes.retry_archive /path/to/original.png
```

This uploads the unchanged PNG and reads its embedded archive description and
album. If upload already succeeded, add `--asset-id <id>` to retry only metadata.
Results are JSON; a non-zero exit code indicates at least one stage errored
(an `errors` entry in the report). `unconfirmed` album states are not
treated as errors and produce exit code 0 — they signal "check whether a
proxy is stripping the response body" rather than a failed archive.

Recovery needs the original file. A PNG with no embedded graph, an unreadable
one, or one holding several archive nodes is refused with a message on stderr
and exit code 1 rather than archiving with no metadata.

## Updating the installation

```bash
cd /path/to/ComfyUI/custom_nodes/comfyui-immich
git pull
```

Your `.env` file is preserved — it's in `.gitignore`.

## License

MIT
