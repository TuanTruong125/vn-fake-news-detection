import { createApiClient } from "./api.js";
import { createHistoryStore } from "./history.js";
import { initUI } from "./ui.js";


// Utility function to format error messages for display in the UI.
function formatClientError(error) {
  if (error?.type === "api_error") {
    const code = error.errorCode ? `[${error.errorCode}] ` : "";
    return `${code}${error.message || "API request failed."}`;
  }
  if (error?.type === "network_error") {
    return error.message || "Network error. Please try again.";
  }
  return error?.message || "Unexpected error during prediction.";
}


// Build a stable unique id for history persistence.
function createHistoryId() {
  const timestamp = Date.now().toString(36);
  const suffix = Math.random().toString(36).slice(2, 8);
  return `hist_${timestamp}_${suffix}`;
}


// Convert a runtime error into a serializable object for history storage.
function normalizeHistoryError(error) {
  if (!error || typeof error !== "object") {
    return {
      type: "unknown_error",
      status: null,
      errorCode: null,
      message: String(error ?? "Unexpected error."),
      raw: null,
    };
  }
  return {
    type: error.type ?? "unknown_error",
    status: error.status ?? null,
    errorCode: error.errorCode ?? null,
    message: error.message ?? error.detail ?? "Unexpected error.",
    detail: error.detail ?? null,
    raw: error.raw ?? null,
  };
}


// Build one history item object from a request + result/error pair.
function buildHistoryItem({ request, response, error, status, latencyMs }) {
  return {
    id: createHistoryId(),
    created_at: new Date().toISOString(),
    request: {
      text: String(request?.text ?? ""),
      model_family: String(request?.model_family ?? "").toLowerCase(),
      run_id: String(request?.run_id ?? "").trim(),
      content_type: String(request?.content_type ?? "").toLowerCase(),
      top_k: Number.isFinite(Number(request?.top_k)) ? Number(request.top_k) : null,
      return_explanation: Boolean(request?.return_explanation),
    },
    response: status === "success" ? response ?? null : null,
    error: status === "error" ? error ?? null : null,
    meta: {
      status,
      latency_ms: Number.isFinite(Number(latencyMs)) ? Math.round(Number(latencyMs)) : null,
    },
  };
}


// Main bootstrap function to initialize the application, set up event listeners, and handle interactions between the UI and API client.
function bootstrap() {
  const ui = initUI();
  const api = createApiClient();
  const historyStore = createHistoryStore();

  ui.setRunLoadingState("Loading available runs...");
  api.fetchRuns()
    .then((runs) => {
      ui.loadRunOptions(runs);
    })
    .catch((error) => {
      ui.setRunLoadingState("Failed to load runs.");
      ui.setFormError(formatClientError(error));
    });

  ui.renderHistoryList(historyStore.getAll());

  ui.elements.modelFamily?.addEventListener("change", () => {
    ui.refreshRunOptions();
    ui.clearFieldErrors();
    ui.setFormError("");
  });

  ui.elements.historyList?.addEventListener("click", (event) => {
    const itemNode = event.target.closest?.(".history-item");
    if (!itemNode) {
      return;
    }
    const item = historyStore.getById(itemNode.dataset.id);
    if (item) {
      ui.applyHistoryItem(item);
    }
  });

  ui.elements.historyList?.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    const itemNode = event.target.closest?.(".history-item");
    if (!itemNode) {
      return;
    }
    event.preventDefault();
    const item = historyStore.getById(itemNode.dataset.id);
    if (item) {
      ui.applyHistoryItem(item);
    }
  });

  document.getElementById("history-clear-btn")?.addEventListener("click", () => {
    if (!historyStore.getAll().length) {
      return;
    }
    const confirmed = window.confirm("Clear all prediction history?");
    if (!confirmed) {
      return;
    }
    historyStore.clear();
    ui.renderHistoryList([]);
  });

  ui.elements.form?.addEventListener("submit", async (event) => {
    event.preventDefault();
    ui.clearFieldErrors();
    ui.setFormError("");

    const payload = ui.getPayload();
    const errors = ui.validatePayload(payload);
    if (Object.keys(errors).length > 0) {
      for (const [field, message] of Object.entries(errors)) {
        ui.setFieldError(field, message);
      }
      return;
    }

    ui.setLoading(true);
    const startedAt = performance.now();
    try {
      const result = await api.predict({
        text: payload.text,
        model_family: payload.model_family,
        run_id: payload.run_id,
        content_type: payload.content_type,
        top_k: payload.top_k,
        return_explanation: payload.return_explanation,
      });
      ui.renderResult(result);
      historyStore.add(
        buildHistoryItem({
          request: payload,
          response: result,
          status: "success",
          latencyMs: performance.now() - startedAt,
        }),
      );
      ui.renderHistoryList(historyStore.getAll());
    } catch (error) {
      const message = formatClientError(error);
      ui.setFormError(message);
      ui.renderError(error);
      historyStore.add(
        buildHistoryItem({
          request: payload,
          error: normalizeHistoryError(error),
          status: "error",
          latencyMs: performance.now() - startedAt,
        }),
      );
      ui.renderHistoryList(historyStore.getAll());
    } finally {
      ui.setLoading(false);
    }
  });

  ui.setStatus("ready");
  window.app = {
    ui,
    api,
    historyStore,
  };
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootstrap);
} else {
  bootstrap();
}
