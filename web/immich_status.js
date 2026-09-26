// Quick Immich status from the "Save to Immich" node's right-click menu:
// "Immich: connection status" or "Immich: test connection". Nothing here changes
// configuration; edit it in Settings -> Immich (immich_settings.js).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const MESSAGES = {
  unauthorized: "Immich rejected the API key (401). Check it in Settings → Immich.",
  forbidden: "The key was accepted but is restricted (403). Uploads may still work if it can upload, read, update and add to albums.",
  unreachable: "Could not reach the Immich URL from the ComfyUI machine.",
  timeout: "Immich did not answer in time.",
  redirect_refused: "Immich answered with a redirect. Save the final address (for example https://) in Settings → Immich.",
  bad_response: "Immich answered, but not as expected.",
  not_configured: "Set the Immich URL and API key in Settings → Immich.",
  rate_limited: "Please wait a few seconds before testing again.",
};

function notify(severity, summary, detail) {
  const toast = app.extensionManager?.toast;
  if (toast?.add) {
    toast.add({ severity, summary, detail, life: 10000 });
  } else {
    window.alert(`${summary}\n\n${detail}`);
  }
}

function describe(status) {
  return [
    `Server: ${status.url ?? "not set"} (from ${status.source.url})`,
    `API key: ${status.key_set ? "set" : "not set"} (from ${status.source.key})`,
    `Saved in: ${status.config_location}. Change it in Settings → Immich.`,
  ].join("\n");
}

async function showStatus() {
  const status = await (await api.fetchApi("/immich/status")).json();
  const ready = status.url && status.key_set;
  notify(ready ? "info" : "warn", "Immich status", describe(status));
}

async function testConnection() {
  const response = await api.fetchApi("/immich/test", { method: "POST" });
  const result = await response.json();
  if (result.ok) {
    notify("success", "Immich connection", "Connected, and the API key works.");
  } else {
    const code = result.status ? ` (HTTP ${result.status})` : "";
    notify("error", "Immich connection", (MESSAGES[result.error] ?? result.error) + code);
  }
}

function guarded(fn) {
  return () => fn().catch((err) => notify("error", "Immich", String(err)));
}

app.registerExtension({
  name: "comfyui-immich.status",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "SaveToImmich") return;
    const original = nodeType.prototype.getExtraMenuOptions;
    nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
      const result = original?.apply(this, arguments);
      options.push(
        { content: "Immich: connection status", callback: guarded(showStatus) },
        { content: "Immich: test connection", callback: guarded(testConnection) },
      );
      return result;
    };
  },
});
