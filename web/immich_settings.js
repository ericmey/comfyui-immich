// Immich connection settings, editable from ComfyUI's Settings dialog.
//
// Nothing here is stored in ComfyUI's own settings file: that file can be read
// back through ComfyUI's API, and it would expose the API key. The form saves to
// this node's route, which writes ComfyUI's user directory. The key is
// write-only: the server never sends it back, so this form only shows "set" or
// "not set".
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function el(tag, props = {}, ...children) {
  const node = Object.assign(document.createElement(tag), props);
  node.append(...children);
  return node;
}

const INPUT_STYLE =
  "width:100%;box-sizing:border-box;padding:6px 8px;border-radius:6px;" +
  "background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color)";
const BUTTON_STYLE =
  "padding:6px 12px;border-radius:6px;cursor:pointer;" +
  "background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color)";

function field(label, input, note) {
  const wrap = el(
    "label",
    { style: "display:block;margin:0.5em 0" },
    el("div", { textContent: label, style: "font-weight:600" }),
    input,
  );
  if (note) wrap.append(el("div", { textContent: note, style: "opacity:0.7;font-size:0.85em" }));
  return wrap;
}

const ERRORS = {
  cross_origin: "Refused: settings can only be saved from this ComfyUI page.",
  confirm_url_change: "Changing the Immich URL needs confirmation.",
  key_outside_panel:
    "Your API key is set outside this panel (an environment variable or the node folder's .env). " +
    "The panel cannot remove it or change the server it is sent to; change both where the key is set.",
  invalid_url: "The Immich URL must be an http(s) URL without a username, password, query or fragment.",
  invalid_api_key: "The API key cannot be blank. Use Clear key to remove it.",
  invalid_value: "Values must be a single line.",
  no_user_directory: "ComfyUI's user directory is unavailable, so settings cannot be saved.",
};

const TEST_MESSAGES = {
  unauthorized: "Immich rejected the API key (401).",
  forbidden: "The key was accepted but is restricted (403). Uploads may still work if it can upload, read, update and add to albums.",
  unreachable: "Could not reach the Immich URL from the ComfyUI machine.",
  timeout: "Immich did not answer in time.",
  redirect_refused: "Immich answered with a redirect. Save the final address (for example https://).",
  bad_response: "Immich answered, but not as expected.",
  not_configured: "Save an Immich URL and API key first.",
};

function settingsPanel() {
  const row = el("tr");
  const cell = el("td", { colSpan: 2 });

  const status = el("div", { textContent: "Loading…" });
  const url = el("input", { type: "url", placeholder: "https://immich.example.com", style: INPUT_STYLE });
  const key = el("input", { type: "password", autocomplete: "new-password", style: INPUT_STYLE });
  const clearKey = el("button", { type: "button", textContent: "Clear key", style: BUTTON_STYLE });
  const save = el("button", { type: "button", textContent: "Save", style: BUTTON_STYLE });
  const test = el("button", { type: "button", textContent: "Test connection", style: BUTTON_STYLE });
  const message = el("div", { style: "margin-top:0.5em" });
  let saved = { url: "" };

  function show(data) {
    saved = data;
    url.value = data.url || "";
    key.value = "";
    key.placeholder = data.key_set ? "•••••••• (set; type a new key to replace it)" : "not set";
    const env = Object.entries(data.shadowed || {})
      .filter(([, on]) => on)
      .map(([name]) => (name === "url" ? "IMMICH_URL" : "IMMICH_API_KEY"));
    status.textContent = env.length
      ? `Overridden by environment variables: ${env.join(", ")}. Values saved here apply only where no variable is set.`
      : `Saved in ${data.config_location}.`;
    save.disabled = data.writable === false;
  }

  async function refresh() {
    try {
      const response = await api.fetchApi("/immich/status");
      if (!response.ok) throw new Error("status unavailable");
      show(await response.json());
    } catch {
      status.textContent = "Immich status unavailable; check the server log.";
    }
  }

  async function post(body) {
    const response = await api.fetchApi("/immich/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(ERRORS[data.error] || `Save failed (${data.error || response.status}).`);
    return data;
  }

  save.addEventListener("click", async () => {
    const body = {};
    const newUrl = url.value.trim().replace(/\/+$/, "").replace(/\/api$/, "");
    if (newUrl !== (saved.url || "")) {
      const ok = window.confirm(
        `Change the Immich URL to "${newUrl || "(none)"}"?\n\n` +
          "The saved API key will be cleared unless you entered a new key, " +
          "so your key is never sent to a server you did not choose.",
      );
      if (!ok) return;
      body.url = newUrl;
      body.confirm_url_change = true;
    }
    if (key.value.trim()) body.api_key = key.value.trim();
    if (!Object.keys(body).length) {
      message.textContent = "Nothing to save.";
      return;
    }
    save.disabled = true;
    message.textContent = "Saving…";
    try {
      show(await post(body));
      message.textContent = "Saved.";
    } catch (err) {
      message.textContent = err.message;
    } finally {
      key.value = "";
      save.disabled = saved.writable === false;
    }
  });

  clearKey.addEventListener("click", async () => {
    if (!window.confirm("Remove the saved Immich API key?")) return;
    try {
      show(await post({ clear_api_key: true }));
      message.textContent = "Key cleared.";
    } catch (err) {
      message.textContent = err.message;
    }
  });

  test.addEventListener("click", async () => {
    test.disabled = true;
    message.textContent = "Testing saved settings…";
    try {
      const response = await api.fetchApi("/immich/test", { method: "POST" });
      const data = await response.json();
      message.textContent =
        response.status === 429
          ? "Please wait five seconds before testing again."
          : data.ok
            ? `Connected, and the API key works (HTTP ${data.status}).`
            : `${TEST_MESSAGES[data.error] || `Connection failed: ${data.error || "unknown"}`}${data.status ? ` (HTTP ${data.status})` : ""}`;
    } catch {
      message.textContent = "Connection test unavailable; check the server log.";
    } finally {
      test.disabled = false;
    }
  });

  cell.append(
    status,
    field("Immich URL", url, "The address you open Immich at, without /api."),
    field("API key", key, "Write-only: it is saved on the server and never shown again."),
    clearKey,
    el("div", { style: "display:flex;gap:0.5em;margin-top:0.5em" }, save, test),
    message,
    el("div", {
      textContent: "Anyone who can use this ComfyUI page can change these settings. Keep ComfyUI private.",
      style: "opacity:0.7;font-size:0.85em;margin-top:0.5em",
    }),
  );
  row.append(cell);
  void refresh();
  return row;
}

app.registerExtension({
  name: "comfyui-immich.settings",
  setup() {
    app.ui.settings.addSetting({
      id: "Immich.Connection",
      name: "Immich connection",
      type: settingsPanel,
      defaultValue: null,
    });
  },
});
