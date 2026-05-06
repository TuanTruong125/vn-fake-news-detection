const HISTORY_STORAGE_KEY = "vnfnd.predict.history";
const MAX_HISTORY_ITEMS = 50;


// Create a history store that manages prediction history in localStorage.
export function createHistoryStore() {

  // Generate a unique id for each history item.
  function createHistoryId() {
    const timestamp = Date.now().toString(36);
    const suffix = Math.random().toString(36).slice(2, 8);
    return `hist_${timestamp}_${suffix}`;
  }


  // Check if a value is a plain object (not null, not array).
  function isPlainObject(value) {
    return Boolean(value) && typeof value === "object" && !Array.isArray(value);
  }


  // Normalize a raw request object into a consistent format.
  function normalizeRequest(request) {
    if (!isPlainObject(request)) {
      return null;
    }
    const text = String(request.text ?? "");
    const modelFamily = String(request.model_family ?? "").toLowerCase().trim();
    const runId = String(request.run_id ?? "").trim();
    const contentType = String(request.content_type ?? "").toLowerCase().trim();
    const topK = Number(request.top_k);
    return {
      text,
      model_family: modelFamily,
      run_id: runId,
      content_type: contentType,
      top_k: Number.isFinite(topK) ? topK : null,
      return_explanation: Boolean(request.return_explanation),
    };
  }


  // Normalize history item error into a consistent format for storage and display.
  function normalizeHistoryItem(item) {
    if (!isPlainObject(item)) {
      return null;
    }

    const request = normalizeRequest(item.request);
    const status = String(item?.meta?.status ?? "").toLowerCase();
    if (!request || (status !== "success" && status !== "error")) {
      return null;
    }

    const response = status === "success" && isPlainObject(item.response) ? item.response : null;
    const error = status === "error" && isPlainObject(item.error) ? item.error : null;
    const latency = Number(item?.meta?.latency_ms);

    return {
      id: String(item.id ?? createHistoryId()),
      created_at: String(item.created_at ?? new Date().toISOString()),
      request,
      response,
      error,
      meta: {
        status,
        latency_ms: Number.isFinite(latency) ? latency : null,
      },
    };
  }


  // Safely parse one history payload from localStorage.
  function getAll() {
    try {
      const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
      if (!raw) {
        return [];
      }
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        return [];
      }
      return parsed.map(normalizeHistoryItem).filter(Boolean).slice(0, MAX_HISTORY_ITEMS);
    } catch {
      return [];
    }
  }


  // Persist a bounded history list.
  function setAll(items) {
    const normalized = Array.isArray(items)
      ? items.map(normalizeHistoryItem).filter(Boolean).slice(0, MAX_HISTORY_ITEMS)
      : [];
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(normalized));
    } catch {
      // Ignore storage quota / availability failures.
    }
    return normalized;
  }


  // Add one history item to the beginning of the list.
  function add(item) {
    const normalizedItem = normalizeHistoryItem(item);
    if (!normalizedItem) {
      return null;
    }
    const current = getAll().filter((entry) => String(entry.id ?? "") !== normalizedItem.id);
    const next = [normalizedItem, ...current].slice(0, MAX_HISTORY_ITEMS);
    setAll(next);
    return normalizedItem;
  }


  // Find one history item by id.
  function getById(id) {
    if (!id) {
      return null;
    }
    return getAll().find((item) => String(item.id ?? "") === String(id)) ?? null;
  }


  // Remove one history item by id.
  function remove(id) {
    if (!id) {
      return [];
    }
    const next = getAll().filter((item) => String(item.id ?? "") !== String(id));
    return setAll(next);
  }


  // Clear all history items.
  function clear() {
    try {
      localStorage.removeItem(HISTORY_STORAGE_KEY);
    } catch {
      // Ignore storage failures.
    }
  }

  return {
    key: HISTORY_STORAGE_KEY,
    maxItems: MAX_HISTORY_ITEMS,
    getAll,
    setAll,
    add,
    getById,
    remove,
    clear,
  };
}
