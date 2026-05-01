import { createApiClient } from "./api.js";
import { createHistoryStore } from "./history.js";
import { initUI } from "./ui.js";

function bootstrap() {
  const ui = initUI();
  const api = createApiClient();
  const historyStore = createHistoryStore();

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
