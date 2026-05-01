export function initUI() {
  const elements = {
    main: document.getElementById("app-main"),
    predictPanel: document.getElementById("predict-panel"),
    resultPanel: document.getElementById("result-panel"),
    historyPanel: document.getElementById("history-panel"),
  };

  return {
    elements,
    setStatus(message) {
      if (!elements.main) {
        return;
      }
      elements.main.dataset.status = message;
    },
  };
}
