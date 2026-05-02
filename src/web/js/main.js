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
  return "Unexpected error during prediction.";
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

  ui.elements.modelFamily?.addEventListener("change", () => {
    ui.refreshRunOptions();
    ui.clearFieldErrors();
    ui.setFormError("");
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
    } catch (error) {
      const message = formatClientError(error);
      ui.setFormError(message);
      ui.renderError(message);
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
