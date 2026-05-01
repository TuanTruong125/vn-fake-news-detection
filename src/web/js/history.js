const HISTORY_STORAGE_KEY = "vnfnd.predict.history";
const MAX_HISTORY_ITEMS = 20;

export function createHistoryStore() {
  function getAll() {
    try {
      const raw = localStorage.getItem(HISTORY_STORAGE_KEY);
      if (!raw) {
        return [];
      }
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  function setAll(items) {
    const normalized = Array.isArray(items) ? items.slice(0, MAX_HISTORY_ITEMS) : [];
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(normalized));
  }

  function clear() {
    localStorage.removeItem(HISTORY_STORAGE_KEY);
  }

  return {
    key: HISTORY_STORAGE_KEY,
    maxItems: MAX_HISTORY_ITEMS,
    getAll,
    setAll,
    clear,
  };
}
