const VALID_FAMILIES = new Set(["ml", "dl"]);
const VALID_CONTENT_TYPES = new Set(["news", "social"]);


// Utility function to normalize input text by unescaping common escape sequences and Unicode characters.
function normalizeInputText(value) {
  let text = String(value ?? "")
    .replace(/\r\n/g, "\n")
    .replace(/\r/g, "\n")
    .replace(/\\n/g, "\n")
    .replace(/\\r/g, "\r")
    .replace(/\\t/g, "\t")
    .replace(/\\"/g, '"');
  text = text.replace(/\\u([0-9a-fA-F]{4})/g, (_, hex) => String.fromCharCode(Number.parseInt(hex, 16)));
  return text;
}


// Initialize tab switching for Result/History sections
function initTabSwitching() {
  const tabButtons = document.querySelectorAll(".tab-button");
  const tabContents = document.querySelectorAll(".tab-content");

  // Animates the transition of tab content when switching between tabs.
  function animateTabContent(content, entering) {
    if (!content || typeof content.animate !== "function") {
      return;
    }
    const animation = content.animate(
      entering
        ? [
            { opacity: 0, transform: "translateY(8px) scale(0.99)" },
            { opacity: 1, transform: "translateY(0) scale(1)" },
          ]
        : [
            { opacity: 1, transform: "translateY(0) scale(1)" },
            { opacity: 0, transform: "translateY(-6px) scale(0.985)" },
          ],
      {
        duration: entering ? 220 : 160,
        easing: "cubic-bezier(0.2, 0.8, 0.2, 1)",
        fill: "both",
      },
    );

    if (!entering) {
      animation.onfinish = () => {
        content.hidden = true;
      };
    }
  }

  tabButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const tabId = button.getAttribute("data-tab");

      tabButtons.forEach((btn) => {
        btn.classList.remove("active");
        btn.setAttribute("aria-selected", "false");
      });
      tabContents.forEach((content) => {
        content.classList.remove("active");
        if (!content.hidden) {
          animateTabContent(content, false);
        }
      });

      button.classList.add("active");
      button.setAttribute("aria-selected", "true");
      const activeContent = document.getElementById(tabId);
      if (activeContent) {
        activeContent.classList.add("active");
        activeContent.hidden = false;
        animateTabContent(activeContent, true);
      }
    });
  });
}


// Factory function to create an API client with configurable base URL and timeout.
export function initUI() {
  
  // Initialize tabs first
  initTabSwitching();

  const elements = {
    main: document.getElementById("app-main"),
    form: document.getElementById("predict-form"),
    formError: document.getElementById("form-error"),
    text: document.getElementById("text"),
    modelFamily: document.getElementById("model_family"),
    modelFamilyDropdown: document.getElementById("model-family-dropdown"),
    modelFamilyDropdownToggle: document.getElementById("model-family-dropdown-toggle"),
    modelFamilyDropdownValue: document.getElementById("model-family-dropdown-value"),
    modelFamilyDropdownMenu: document.getElementById("model-family-dropdown-menu"),
    contentType: document.getElementById("content_type"),
    contentTypeDropdown: document.getElementById("content-type-dropdown"),
    contentTypeDropdownToggle: document.getElementById("content-type-dropdown-toggle"),
    contentTypeDropdownValue: document.getElementById("content-type-dropdown-value"),
    contentTypeDropdownMenu: document.getElementById("content-type-dropdown-menu"),
    runId: document.getElementById("run_id"),
    runSelectBestTag: document.getElementById("run-select-best-tag"),
    runDropdown: document.getElementById("run-dropdown"),
    runDropdownToggle: document.getElementById("run-dropdown-toggle"),
    runDropdownValue: document.getElementById("run-dropdown-value"),
    runDropdownMenu: document.getElementById("run-dropdown-menu"),
    topK: document.getElementById("top_k"),
    returnExplanation: document.getElementById("return_explanation"),
    runStatus: document.getElementById("run-status"),
    predictButton: document.getElementById("predict-button"),
    buttonLabel: document.querySelector("#predict-button .btn-label"),
    resultNote: document.getElementById("result-note"),
    resultJson: document.getElementById("result-json"),
    resultSummary: document.getElementById("result-summary"),
    resultBadge: document.getElementById("result-badge"),
    resultConfidence: document.getElementById("result-confidence"),
    resultMetricGrid: document.getElementById("result-metric-grid"),
    resultModelGrid: document.getElementById("result-model-grid"),
    resultWarnings: document.getElementById("result-warnings"),
    resultErrorBox: document.getElementById("result-error-box"),
    copyJsonButton: document.getElementById("copy-json-btn"),
    copyJsonState: document.getElementById("copy-json-state"),
    resultTab: document.getElementById("result-tab"),
    historyTab: document.getElementById("history-tab"),
    historyList: document.getElementById("history-list"),
    fieldErrors: {
      text: document.getElementById("text-error"),
      model_family: document.getElementById("model-family-error"),
      content_type: document.getElementById("content-type-error"),
      run_id: document.getElementById("run-id-error"),
      top_k: document.getElementById("top-k-error"),
    },
  };

  const state = {
    allRuns: [],
    isLoading: false,
    latestResultJson: "",
    copyStateTimer: null,
    allHistoryItems: [],
    openDropdownKey: null,
    modelInfoExpanded: {},
  };


  // Ensure explanation panel exists in Result tab.
  function ensureExplanationPanel() {
    if (!elements.resultTab) {
      return null;
    }

    let panel = document.getElementById("explain-panel");
    if (!panel) {
      panel = document.createElement("section");
      panel.id = "explain-panel";
      panel.className = "explain-panel";
      panel.hidden = true;
      panel.innerHTML = `
        <div class="explain-head">
          <h3>Explanation</h3>
          <p id="explain-status" class="explain-status"></p>
        </div>
        <div id="explain-empty" class="explain-empty" hidden></div>
        <div id="explain-content" hidden>
          <div class="explain-columns">
            <div class="explain-col fake">
              <h4>Towards FAKE</h4>
              <div id="explain-fake-list" class="explain-list"></div>
            </div>
            <div class="explain-col real">
              <h4>Towards REAL</h4>
              <div id="explain-real-list" class="explain-list"></div>
            </div>
          </div>
          <div class="explain-decomposition">
            <h4>Decomposition</h4>
            <div id="explain-decomposition-grid" class="explain-decomposition-grid"></div>
            <p id="explain-decomposition-note" class="explain-decomposition-note"></p>
          </div>
        </div>
      `;
      const anchor = elements.resultErrorBox ?? elements.resultJson;
      if (anchor?.parentNode === elements.resultTab) {
        elements.resultTab.insertBefore(panel, anchor);
      } else {
        elements.resultTab.appendChild(panel);
      }
    }
    return {
      panel,
      status: document.getElementById("explain-status"),
      empty: document.getElementById("explain-empty"),
      content: document.getElementById("explain-content"),
      fakeList: document.getElementById("explain-fake-list"),
      realList: document.getElementById("explain-real-list"),
      decompositionGrid: document.getElementById("explain-decomposition-grid"),
      decompositionNote: document.getElementById("explain-decomposition-note"),
    };
  }

  const explanation = ensureExplanationPanel();


  // Ensure history controls exist in History tab.
  function ensureHistoryWidgets() {
    if (!elements.historyTab) {
      return null;
    }
    let root = document.getElementById("history-widgets");
    if (!root) {
      root = document.createElement("div");
      root.id = "history-widgets";
      root.className = "history-widgets";
      root.innerHTML = `
        <div class="history-toolbar">
          <strong>Prediction History</strong>
          <button type="button" id="history-clear-btn" class="history-clear-btn">Clear</button>
        </div>
        <div id="history-empty" class="history-empty" hidden></div>
      `;
      const placeholder = elements.historyList?.parentNode;
      if (placeholder && placeholder.parentNode === elements.historyTab) {
        elements.historyTab.insertBefore(root, placeholder);
      } else {
        elements.historyTab.appendChild(root);
      }
    }
    return {
      root,
      clearButton: document.getElementById("history-clear-btn"),
      empty: document.getElementById("history-empty"),
    };
  }

  const historyWidgets = ensureHistoryWidgets();


  // Set the status message for the main application container.
  function setStatus(message) {
    if (elements.main) {
      elements.main.dataset.status = message;
    }
  }


  // Set the error message for the entire form.
  function setFormError(message) {
    if (!elements.formError) {
      return;
    }
    if (!message) {
      elements.formError.hidden = true;
      elements.formError.textContent = "";
      return;
    }
    elements.formError.hidden = false;
    elements.formError.textContent = message;
  }


  // Clear all field-specific error messages.
  function clearFieldErrors() {
    for (const [field, errorNode] of Object.entries(elements.fieldErrors)) {
      if (!errorNode) {
        continue;
      }
      errorNode.hidden = true;
      errorNode.textContent = "";
      getFieldElement(field)?.classList.remove("is-invalid");
      getFieldElement(field)?.removeAttribute("aria-invalid");
    }
  }


  // Get the HTML element for a specific form field.
  function getFieldElement(field) {
    switch (field) {
      case "text":
        return elements.text;
      case "model_family":
        return elements.modelFamilyDropdownToggle ?? elements.modelFamily;
      case "content_type":
        return elements.contentTypeDropdownToggle ?? elements.contentType;
      case "run_id":
        return elements.runDropdownToggle ?? elements.runId;
      case "top_k":
        return elements.topK;
      default:
        return null;
    }
  }


  // Set the error message for a specific form field.
  function setFieldError(field, message) {
    const node = elements.fieldErrors[field];
    const fieldElement = getFieldElement(field);
    if (!node || !fieldElement) {
      return;
    }
    node.hidden = false;
    node.textContent = message;
    fieldElement.classList.add("is-invalid");
    fieldElement.setAttribute("aria-invalid", "true");
  }


  // Set the loading state of the form, disabling inputs and showing a loading indicator as needed.
  function setLoading(isLoading) {
    state.isLoading = isLoading;
    if (elements.predictButton) {
      elements.predictButton.disabled = isLoading;
    }
    if (elements.buttonLabel) {
      elements.buttonLabel.textContent = isLoading ? "Predicting..." : "Predict";
    }
  }


  // Format confidence value to percentage string.
  function formatPercent(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "-";
    }
    return `${(number * 100).toFixed(2)}%`;
  }


  // Format milliseconds for readable display.
  function formatMs(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "-";
    }
    return `${number.toFixed(2)} ms`;
  }


  // Resolve result label text and CSS class from response.
  function resolveLabel(result) {
    const labelId = Number(result?.label_id);
    const labelText = String(result?.label_text ?? "").trim().toLowerCase();
    if (labelId === 1 || labelText === "fake") {
      return { text: "FAKE", className: "result-badge fake" };
    }
    if (labelId === 0 || labelText === "real") {
      return { text: "REAL", className: "result-badge real" };
    }
    return { text: String(result?.label_text ?? "UNKNOWN").toUpperCase(), className: "result-badge unknown" };
  }

  
  // Format signed numeric values for explanation rows.
  function formatSignedNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "-";
    }
    return `${number >= 0 ? "+" : ""}${number}`;
  }


  // Format nullable numeric values for decomposition fields.
  function formatNullableNumber(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
      return "-";
    }
    return number;
  }


  // Safely render key-value rows to a target container.
  function renderKeyValueGrid(target, rows) {
    if (!target) {
      return;
    }
    target.innerHTML = "";
    for (const row of rows) {
      const item = document.createElement("div");
      item.className = "result-kv-item";
      const key = document.createElement("span");
      key.className = "result-kv-key";
      key.textContent = row.key;
      const value = document.createElement("strong");
      value.className = `result-kv-value ${row.valueClass ?? ""}`.trim();

      if (row.badgeText) {
        value.classList.add("result-kv-value-inline");
        const valueText = document.createElement("span");
        valueText.className = "result-run-id";
        valueText.textContent = row.value;
        const badge = document.createElement("span");
        badge.className = `best-run-tag ${row.badgeClass ?? ""}`.trim();
        badge.textContent = row.badgeText;
        value.appendChild(valueText);
        value.appendChild(badge);
      } else {
        value.textContent = row.value;
      }

      item.appendChild(key);
      item.appendChild(value);
      target.appendChild(item);
    }
  }


  // Check if a run_id is marked as best for a specific model family.
  function isBestRun(runId, modelFamily) {
    const normalizedRunId = String(runId ?? "").trim();
    const normalizedFamily = String(modelFamily ?? "").toLowerCase().trim();
    if (!normalizedRunId) {
      return false;
    }
    return state.allRuns.some((run) => {
      const sameRunId = String(run?.run_id ?? "").trim() === normalizedRunId;
      const sameFamily = normalizedFamily
        ? String(run?.model_family ?? "").toLowerCase().trim() === normalizedFamily
        : true;
      return sameRunId && sameFamily && Boolean(run?.is_best);
    });
  }


  // Get 1-based rank of a run within its model family after sorting (returns null if not found)
  function getRunRank(runId, modelFamily) {
    const normalizedRunId = String(runId ?? "").trim();
    const normalizedFamily = String(modelFamily ?? "").toLowerCase().trim();
    if (!normalizedRunId) return null;
    const filtered = state.allRuns.filter((item) => {
      const sameFamily = normalizedFamily ? String(item?.model_family ?? "").toLowerCase().trim() === normalizedFamily : true;
      return sameFamily;
    });
    if (!filtered.length) return null;
    const sorted = sortRunsByMetrics(filtered);
    const idx = sorted.findIndex((r) => String(r?.run_id ?? "") === normalizedRunId);
    return idx === -1 ? null : idx + 1;
  }


  // Show/hide BEST tag for current Run ID selection.
  function updateRunSelectBestTag(runId, modelFamily) {
    if (!elements.runSelectBestTag) {
      return;
    }
    const family = String(modelFamily ?? "").toLowerCase();
    const filtered = state.allRuns.filter((item) => String(item?.model_family ?? "").toLowerCase() === family);
    if (!filtered.length) {
      elements.runSelectBestTag.hidden = true;
      return;
    }
    const sorted = sortRunsByMetrics(filtered);
    const idx = sorted.findIndex((item) => String(item?.run_id ?? "") === String(runId ?? ""));
    if (idx === -1 || idx >= 3) {
      elements.runSelectBestTag.hidden = true;
      return;
    }
    const rank = idx + 1;
    elements.runSelectBestTag.hidden = false;
    elements.runSelectBestTag.textContent = rank === 1 ? "1st" : rank === 2 ? "2nd" : "3rd";
    elements.runSelectBestTag.className = `run-rank-tag run-rank-${rank} run-select-best-tag`;
  }


  // Build the item list for a custom dropdown field.
  function buildDropdownItems(field) {
    switch (field) {
      case "content_type":
        return [
          { value: "news", label: "News" },
          { value: "social", label: "Social" },
        ];
      case "model_family":
        return [
          { value: "ml", label: "ML" },
          { value: "dl", label: "DL" },
        ];
      default:
        return [];
    }
  }


  // Resolve a human-friendly label for a dropdown value.
  function getDropdownLabel(field, value) {
    const normalizedValue = String(value ?? "").toLowerCase();
    const items = buildDropdownItems(field);
    const matched = items.find((item) => item.value === normalizedValue);
    return matched?.label ?? (normalizedValue ? String(value) : "Select...");
  }


  // Create the DOM nodes for a dropdown option list.
  function renderDropdownMenu(menu, items, selectedValue) {
    if (!menu) {
      return;
    }
    menu.innerHTML = "";
    for (const item of items) {
      const option = document.createElement("button");
      option.type = "button";
      option.className = "run-dropdown-option";
      option.setAttribute("role", "option");
      option.dataset.value = item.value;
      option.setAttribute("aria-selected", item.value === selectedValue ? "true" : "false");
      option.classList.toggle("is-selected", item.value === selectedValue);

      const text = document.createElement("span");
      text.className = "run-dropdown-option-text";
      text.textContent = item.label;
      option.appendChild(text);

      if (item.isBest) {
        const badge = document.createElement("span");
        badge.className = "best-run-tag run-dropdown-option-tag";
        badge.textContent = "BEST";
        option.appendChild(badge);
      }

      menu.appendChild(option);
    }
  }


  // Open or close a custom dropdown controller.
  function setCustomDropdownOpen(controller, open) {
    if (!controller?.toggle || !controller?.menu) {
      return;
    }
    if (open) {
      if (state.openDropdownKey && state.openDropdownKey !== controller.key) {
        const previous = customDropdownControllers[state.openDropdownKey];
        if (previous) {
          setCustomDropdownOpen(previous, false);
        }
      }
      if (state.runDropdownOpen) {
        setRunDropdownOpen(false);
      }
    }
    controller.menu.hidden = !open;
    controller.toggle.setAttribute("aria-expanded", open ? "true" : "false");
    state.openDropdownKey = open ? controller.key : (state.openDropdownKey === controller.key ? null : state.openDropdownKey);
    if (open) {
      const selected = controller.menu.querySelector(".run-dropdown-option.is-selected")
        ?? controller.menu.querySelector(".run-dropdown-option");
      selected?.focus();
    }
  }


  // Sync the visible label and hidden select for a custom dropdown.
  function syncCustomDropdownSelection(controller, value) {
    if (!controller) {
      return;
    }
    const normalizedValue = String(value ?? "").toLowerCase();
    const label = getDropdownLabel(controller.key, normalizedValue);
    if (controller.value) {
      controller.value.textContent = label;
    }
    if (controller.toggle) {
      controller.toggle.classList.toggle("is-placeholder", !normalizedValue);
    }
    if (controller.select) {
      controller.select.value = normalizedValue;
    }
    if (controller.menu) {
      controller.menu.querySelectorAll(".run-dropdown-option").forEach((option) => {
        const isSelected = option.dataset.value === normalizedValue;
        option.setAttribute("aria-selected", isSelected ? "true" : "false");
        option.classList.toggle("is-selected", isSelected);
      });
    }
  }


  // Find the active custom dropdown option to focus based on arrow navigation.
  function focusCustomDropdownOption(controller, direction) {
    if (!controller?.menu) {
      return;
    }
    const options = Array.from(controller.menu.querySelectorAll(".run-dropdown-option"));
    if (!options.length) {
      return;
    }
    const currentIndex = options.findIndex((item) => item === document.activeElement);
    let nextIndex = 0;
    if (direction === "first") {
      nextIndex = 0;
    } else if (direction === "last") {
      nextIndex = options.length - 1;
    } else if (direction === "next") {
      nextIndex = Math.min(options.length - 1, currentIndex < 0 ? 0 : currentIndex + 1);
    } else if (direction === "prev") {
      nextIndex = Math.max(0, currentIndex < 0 ? 0 : currentIndex - 1);
    }
    options[nextIndex]?.focus();
  }


  const customDropdownControllers = {};


  // Initialize a custom dropdown for static option sets such as Content Type and Model Type.
  function initCustomDropdownController(controller) {
    if (!controller?.toggle || !controller?.menu || !controller?.select) {
      return;
    }
    customDropdownControllers[controller.key] = controller;
    const items = buildDropdownItems(controller.key);
    renderDropdownMenu(controller.menu, items, controller.select.value);
    syncCustomDropdownSelection(controller, controller.select.value);

    controller.toggle.addEventListener("click", () => {
      if (controller.toggle.disabled) {
        return;
      }
      const isOpen = state.openDropdownKey === controller.key;
      setCustomDropdownOpen(controller, !isOpen);
    });

    controller.toggle.addEventListener("keydown", (event) => {
      if (controller.toggle.disabled) {
        return;
      }
      if (event.key === "ArrowDown") {
        event.preventDefault();
        setCustomDropdownOpen(controller, true);
        focusCustomDropdownOption(controller, "first");
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        setCustomDropdownOpen(controller, true);
        focusCustomDropdownOption(controller, "last");
      } else if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        setCustomDropdownOpen(controller, !(state.openDropdownKey === controller.key));
      } else if (event.key === "Escape") {
        setCustomDropdownOpen(controller, false);
      }
    });

    controller.menu.addEventListener("click", (event) => {
      const option = event.target.closest?.(".run-dropdown-option");
      if (!option) {
        return;
      }
      controller.select.value = option.dataset.value ?? "";
      syncCustomDropdownSelection(controller, controller.select.value);
      setCustomDropdownOpen(controller, false);
      controller.toggle.focus();
      controller.select.dispatchEvent(new Event("change", { bubbles: true }));
    });

    controller.menu.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        focusCustomDropdownOption(controller, "next");
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        focusCustomDropdownOption(controller, "prev");
      } else if (event.key === "Home") {
        event.preventDefault();
        focusCustomDropdownOption(controller, "first");
      } else if (event.key === "End") {
        event.preventDefault();
        focusCustomDropdownOption(controller, "last");
      } else if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        const option = document.activeElement?.closest?.(".run-dropdown-option");
        if (option) {
          controller.select.value = option.dataset.value ?? "";
          syncCustomDropdownSelection(controller, controller.select.value);
          setCustomDropdownOpen(controller, false);
          controller.toggle.focus();
          controller.select.dispatchEvent(new Event("change", { bubbles: true }));
        }
      } else if (event.key === "Escape") {
        event.preventDefault();
        setCustomDropdownOpen(controller, false);
        controller.toggle.focus();
      }
    });

    controller.toggle.addEventListener("blur", () => {
      window.setTimeout(() => {
        if (!controller.container.contains(document.activeElement)) {
          setCustomDropdownOpen(controller, false);
        }
      }, 0);
    });

    controller.menu.addEventListener("focusin", () => {
      setCustomDropdownOpen(controller, true);
    });
  }


  // Close all custom dropdowns when clicking outside.
  function bindCustomDropdownOutsideClick() {
    document.addEventListener("click", (event) => {
      const target = event.target;
      if (!(target instanceof Node)) {
        return;
      }
      for (const controller of Object.values(customDropdownControllers)) {
        if (state.openDropdownKey === controller.key && !controller.container.contains(target)) {
          setCustomDropdownOpen(controller, false);
        }
      }
      if (state.runDropdownOpen && elements.runDropdown && !elements.runDropdown.contains(target)) {
        setRunDropdownOpen(false);
      }
    });
  }


  // Render the custom run dropdown options and keep the hidden select synchronized.
  function renderRunDropdownOptions(runs, selectedRunId) {
    if (!elements.runId || !elements.runDropdownMenu) {
      return;
    }

    elements.runId.innerHTML = "";
    elements.runDropdownMenu.innerHTML = "";

    if (!Array.isArray(runs) || !runs.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "";
      elements.runId.appendChild(option);

      const empty = document.createElement("div");
      empty.className = "run-dropdown-empty";
      empty.textContent = "No run available";
      elements.runDropdownMenu.appendChild(empty);
      return;
    }

    runs.forEach((run, idx) => {
      const option = document.createElement("option");
      option.value = run.run_id;
      option.textContent = run.run_id;
      elements.runId.appendChild(option);

      const item = document.createElement("button");
      item.type = "button";
      item.className = "run-dropdown-option";
      item.setAttribute("role", "option");
      item.dataset.runId = run.run_id;
      item.dataset.isBest = run.is_best ? "true" : "false";
      const rank = idx + 1;
      item.dataset.rank = String(rank);
      item.setAttribute("aria-selected", run.run_id === selectedRunId ? "true" : "false");

      const text = document.createElement("span");
      text.className = "run-dropdown-option-text";
      text.textContent = run.run_id;
      item.appendChild(text);

      if (rank <= 3) {
        const badge = document.createElement("span");
        badge.className = `run-rank-tag run-rank-${rank} run-dropdown-option-tag`;
        badge.textContent = rank === 1 ? "1st" : rank === 2 ? "2nd" : "3rd";
        item.appendChild(badge);
      }

      elements.runDropdownMenu.appendChild(item);
    });
  }


  // Update the visible run dropdown label and selected state.
  function syncRunDropdownSelection(runId, modelFamily) {
    if (!elements.runDropdownValue) {
      return;
    }
    const normalizedRunId = String(runId ?? "").trim();
    const selectedRun = state.allRuns.find((run) => String(run?.run_id ?? "").trim() === normalizedRunId);
    const labelText = normalizedRunId || "Select model";
    elements.runDropdownValue.textContent = labelText;

    if (elements.runDropdownToggle) {
      elements.runDropdownToggle.classList.toggle("is-placeholder", !normalizedRunId);
      elements.runDropdownToggle.dataset.best = selectedRun?.is_best ? "true" : "false";
    }

    if (elements.runId) {
      elements.runId.value = normalizedRunId;
    }

    if (elements.runSelectBestTag) {
      elements.runSelectBestTag.hidden = !Boolean(selectedRun?.is_best);
    }

    if (elements.runDropdownMenu) {
      const optionButtons = elements.runDropdownMenu.querySelectorAll(".run-dropdown-option");
      optionButtons.forEach((button) => {
        const isSelected = button.dataset.runId === normalizedRunId;
        button.setAttribute("aria-selected", isSelected ? "true" : "false");
        button.classList.toggle("is-selected", isSelected);
      });
    }

    updateRunSelectBestTag(normalizedRunId, modelFamily);
  }


  // Open or close the custom run dropdown menu.
  function setRunDropdownOpen(open) {
    if (!elements.runDropdownMenu || !elements.runDropdownToggle) {
      return;
    }
    if (open) {
      for (const controller of Object.values(customDropdownControllers)) {
        setCustomDropdownOpen(controller, false);
      }
    }
    state.runDropdownOpen = Boolean(open);
    elements.runDropdownMenu.hidden = !state.runDropdownOpen;
    elements.runDropdownToggle.setAttribute("aria-expanded", state.runDropdownOpen ? "true" : "false");
    if (state.runDropdownOpen) {
      const selected = elements.runDropdownMenu.querySelector(".run-dropdown-option.is-selected")
        ?? elements.runDropdownMenu.querySelector(".run-dropdown-option");
      selected?.focus();
    }
  }


  // Select a run_id from the custom dropdown and update dependent UI.
  function selectRunId(runId) {
    if (!elements.runId) {
      return;
    }
    const normalizedRunId = String(runId ?? "").trim();
    elements.runId.value = normalizedRunId;
    syncRunDropdownSelection(normalizedRunId, elements.modelFamily?.value ?? "");
    displayModelInfo(normalizedRunId);
    clearFieldErrors();
    setFormError("");
    setRunDropdownOpen(false);
    elements.runDropdownToggle?.focus();
  }


  // Move focus among run options inside the custom dropdown.
  function focusRunDropdownOption(direction) {
    if (!elements.runDropdownMenu) {
      return;
    }
    const options = Array.from(elements.runDropdownMenu.querySelectorAll(".run-dropdown-option"));
    if (!options.length) {
      return;
    }
    const currentIndex = options.findIndex((item) => item === document.activeElement);
    let nextIndex = 0;
    if (direction === "first") {
      nextIndex = 0;
    } else if (direction === "last") {
      nextIndex = options.length - 1;
    } else if (direction === "next") {
      nextIndex = Math.min(options.length - 1, currentIndex < 0 ? 0 : currentIndex + 1);
    } else if (direction === "prev") {
      nextIndex = Math.max(0, currentIndex < 0 ? 0 : currentIndex - 1);
    }
    options[nextIndex]?.focus();
  }


  // Bind custom dropdown interactions.
  function initRunDropdown() {
    if (!elements.runDropdownToggle || !elements.runDropdownMenu || !elements.runDropdown) {
      return;
    }

    elements.runDropdownToggle.addEventListener("click", () => {
      if (elements.runDropdownToggle.disabled) {
        return;
      }
      setRunDropdownOpen(!state.runDropdownOpen);
    });

    elements.runDropdownToggle.addEventListener("blur", () => {
      window.setTimeout(() => {
        if (!elements.runDropdown.contains(document.activeElement)) {
          setRunDropdownOpen(false);
        }
      }, 0);
    });

    elements.runDropdownToggle.addEventListener("keydown", (event) => {
      if (elements.runDropdownToggle.disabled) {
        return;
      }
      if (event.key === "ArrowDown") {
        event.preventDefault();
        setRunDropdownOpen(true);
        focusRunDropdownOption("first");
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        setRunDropdownOpen(true);
        focusRunDropdownOption("last");
      } else if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        setRunDropdownOpen(!state.runDropdownOpen);
      } else if (event.key === "Escape") {
        setRunDropdownOpen(false);
      }
    });

    elements.runDropdownMenu.addEventListener("click", (event) => {
      const option = event.target.closest?.(".run-dropdown-option");
      if (!option) {
        return;
      }
      selectRunId(option.dataset.runId ?? "");
    });

    elements.runDropdownMenu.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        focusRunDropdownOption("next");
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        focusRunDropdownOption("prev");
      } else if (event.key === "Home") {
        event.preventDefault();
        focusRunDropdownOption("first");
      } else if (event.key === "End") {
        event.preventDefault();
        focusRunDropdownOption("last");
      } else if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        const option = document.activeElement?.closest?.(".run-dropdown-option");
        if (option) {
          selectRunId(option.dataset.runId ?? "");
        }
      } else if (event.key === "Escape") {
        event.preventDefault();
        setRunDropdownOpen(false);
        elements.runDropdownToggle.focus();
      }
    });

    elements.runDropdownMenu.addEventListener("focusin", () => {
      setRunDropdownOpen(true);
    });

    document.addEventListener("click", (event) => {
      if (!state.runDropdownOpen) {
        return;
      }
      const target = event.target;
      if (target instanceof Node && !elements.runDropdown.contains(target)) {
        setRunDropdownOpen(false);
      }
    });
  }


  // Prevent keyboard typing in Top K; allow only spinner controls and pointer interactions.
  function initTopKSpinnerOnlyInput() {
    if (!elements.topK) {
      return;
    }
    elements.topK.addEventListener("keydown", (event) => {
      event.preventDefault();
    });
    elements.topK.addEventListener("paste", (event) => {
      event.preventDefault();
    });
    elements.topK.addEventListener("drop", (event) => {
      event.preventDefault();
    });
  }


  // Update copy button feedback state.
  function setCopyState(message) {
    if (!elements.copyJsonState) {
      return;
    }
    elements.copyJsonState.textContent = message;
    if (state.copyStateTimer) {
      clearTimeout(state.copyStateTimer);
      state.copyStateTimer = null;
    }
    if (!message) {
      return;
    }
    state.copyStateTimer = setTimeout(() => {
      if (elements.copyJsonState) {
        elements.copyJsonState.textContent = "";
      }
      state.copyStateTimer = null;
    }, 1600);
  }


  // Render explanation feature list for one direction.
  function renderExplainFeatureList(target, features, directionClass) {
    if (!target) {
      return;
    }
    target.innerHTML = "";
    if (!Array.isArray(features) || !features.length) {
      const empty = document.createElement("div");
      empty.className = "explain-list-empty";
      empty.textContent = "No feature contributions returned.";
      target.appendChild(empty);
      return;
    }

    for (const feature of features) {
      const row = document.createElement("div");
      row.className = `explain-item ${directionClass}`;
      const featureName = document.createElement("div");
      featureName.className = "explain-item-feature";
      featureName.textContent = String(feature?.feature ?? "-");
      const featureMeta = document.createElement("div");
      featureMeta.className = "explain-item-meta";
      featureMeta.textContent = `contribution ${formatSignedNumber(feature?.contribution)} | weight ${formatSignedNumber(feature?.weight)} | value ${formatNullableNumber(feature?.value)}`;
      row.appendChild(featureName);
      row.appendChild(featureMeta);
      target.appendChild(row);
    }
  }


  // Render explanation decomposition block.
  function renderExplainDecomposition(data) {
    if (!explanation?.decompositionGrid || !explanation?.decompositionNote) {
      return;
    }
    explanation.decompositionGrid.innerHTML = "";
    explanation.decompositionNote.textContent = "";

    if (!data || typeof data !== "object") {
      const empty = document.createElement("div");
      empty.className = "explain-list-empty";
      empty.textContent = "Decomposition is not available.";
      explanation.decompositionGrid.appendChild(empty);
      return;
    }

    const rows = [
      { key: "Sum Positive", value: formatNullableNumber(data.sum_positive_contrib) },
      { key: "Sum Negative", value: formatNullableNumber(data.sum_negative_contrib) },
      { key: "Sum Total", value: formatNullableNumber(data.sum_total_contrib) },
      { key: "Intercept", value: formatNullableNumber(data.intercept) },
      { key: "Estimated Score", value: formatNullableNumber(data.estimated_decision_score) },
      { key: "Raw Decision", value: formatNullableNumber(data.raw_decision_score) },
      { key: "Decision Gap", value: formatNullableNumber(data.decision_score_gap) },
    ];

    for (const row of rows) {
      const item = document.createElement("div");
      item.className = "explain-decomp-item";
      const key = document.createElement("span");
      key.className = "explain-decomp-key";
      key.textContent = row.key;
      const value = document.createElement("strong");
      value.className = "explain-decomp-value";
      value.textContent = row.value;
      item.appendChild(key);
      item.appendChild(value);
      explanation.decompositionGrid.appendChild(item);
    }

    if (typeof data.approximate_score_alignment_note === "string" && data.approximate_score_alignment_note.trim()) {
      explanation.decompositionNote.textContent = data.approximate_score_alignment_note;
    }
  }


  // Render explanation panel based on inference response.
  function renderExplanation(result) {
    if (!explanation?.panel) {
      return;
    }
    explanation.panel.hidden = false;

    const available = Boolean(result?.explanation_available);
    const reason = String(result?.explanation_reason ?? "").trim();

    if (explanation.status) {
      explanation.status.className = available ? "explain-status available" : "explain-status unavailable";
      explanation.status.textContent = available ? "Explanation available." : "Explanation unavailable.";
    }

    if (!available) {
      if (explanation.empty) {
        explanation.empty.hidden = false;
        explanation.empty.textContent = reason || "This model/run does not provide explanation for the current request.";
      }
      if (explanation.content) {
        explanation.content.hidden = true;
      }
      return;
    }

    if (explanation.empty) {
      explanation.empty.hidden = true;
      explanation.empty.textContent = "";
    }
    if (explanation.content) {
      explanation.content.hidden = false;
    }
    renderExplainFeatureList(explanation.fakeList, result?.top_features_towards_fake, "fake");
    renderExplainFeatureList(explanation.realList, result?.top_features_towards_real, "real");
    renderExplainDecomposition(result?.explanation_decomposition);
  }

  
  // Reset explanation panel to neutral state.
  function resetExplanation() {
    if (!explanation?.panel) {
      return;
    }
    explanation.panel.hidden = true;
    if (explanation.status) {
      explanation.status.textContent = "";
      explanation.status.className = "explain-status";
    }
    if (explanation.empty) {
      explanation.empty.hidden = true;
      explanation.empty.textContent = "";
    }
    if (explanation.content) {
      explanation.content.hidden = true;
    }
    if (explanation.fakeList) {
      explanation.fakeList.innerHTML = "";
    }
    if (explanation.realList) {
      explanation.realList.innerHTML = "";
    }
    if (explanation.decompositionGrid) {
      explanation.decompositionGrid.innerHTML = "";
    }
    if (explanation.decompositionNote) {
      explanation.decompositionNote.textContent = "";
    }
  }


  // Switch UI to the Result tab.
  function activateResultTab() {
    const button = document.querySelector('.tab-button[data-tab="result-tab"]');
    if (button) {
      button.click();
    }
  }


  // Render one compact history list item.
  function createHistoryItemNode(item) {
    const container = document.createElement("article");
    container.className = `history-item ${item?.meta?.status === "error" ? "error" : "success"}`;
    container.dataset.id = String(item?.id ?? "");
    container.tabIndex = 0;
    container.setAttribute("role", "button");

    const statusText = item?.meta?.status === "error" ? "Error" : "Success";
    const modelFamily = item?.request?.model_family ? String(item.request.model_family).toUpperCase() : "-";
    const modelFamilyLower = item?.request?.model_family ? String(item.request.model_family).toLowerCase() : "";
    const runId = item?.request?.run_id ? String(item.request.run_id) : "-";
    const bestRun = isBestRun(runId, modelFamilyLower);
    const timestamp = item?.created_at ? new Date(item.created_at).toLocaleString() : "-";
    const preview = String(item?.request?.text ?? "").replace(/\s+/g, " ").trim();
    const previewText = preview.length > 110 ? `${preview.slice(0, 110)}...` : preview || "No text preview.";

    const labelText = item?.meta?.status === "success" && item?.response?.label_text
      ? String(item.response.label_text).toUpperCase()
      : "-";
    const confidence = item?.meta?.status === "success" && Number.isFinite(Number(item?.response?.confidence))
      ? `${(Number(item.response.confidence) * 100).toFixed(2)}%`
      : "-";

    container.innerHTML = `
      <div class="history-item-head">
        <strong class="history-item-family"></strong>
        <span class="history-item-time"></span>
      </div>
      <div class="history-item-main">
        <span class="history-item-run"></span>
        <span class="history-item-status ${item?.meta?.status === "error" ? "error" : "success"}"></span>
      </div>
      <div class="history-item-preview"></div>
      <div class="history-item-foot">
        <span class="history-item-summary"></span>
      </div>
    `;
    const familyNode = container.querySelector(".history-item-family");
    const timeNode = container.querySelector(".history-item-time");
    const runNode = container.querySelector(".history-item-run");
    const statusNode = container.querySelector(".history-item-status");
    const previewNode = container.querySelector(".history-item-preview");
    const summaryNode = container.querySelector(".history-item-summary");
    if (familyNode) {
      familyNode.textContent = modelFamily;
    }
    if (timeNode) {
      timeNode.textContent = timestamp;
    }
    if (runNode) {
      runNode.innerHTML = "";
      const runIdText = document.createElement("span");
      runIdText.className = "history-run-id";
      runIdText.textContent = runId;
      runNode.appendChild(runIdText);

      const rank = getRunRank(runId, modelFamilyLower);
      if (rank && rank <= 3) {
        const rankTag = document.createElement("span");
        rankTag.className = `run-rank-tag history-run-rank run-rank-${rank}`;
        rankTag.textContent = rank === 1 ? "1st" : rank === 2 ? "2nd" : "3rd";
        runNode.appendChild(rankTag);
      }
    }
    if (statusNode) {
      statusNode.textContent = statusText;
    }
    if (previewNode) {
      previewNode.textContent = previewText;
    }
    if (summaryNode) {
      summaryNode.textContent = `${labelText} • ${confidence}`;
    }
    return container;
  }


  // Render full history list from state.
  function renderHistoryList(items) {
    state.allHistoryItems = Array.isArray(items) ? items : [];
    if (!elements.historyList) {
      return;
    }
    elements.historyList.innerHTML = "";

    if (historyWidgets?.empty) {
      historyWidgets.empty.hidden = state.allHistoryItems.length > 0;
      historyWidgets.empty.textContent = state.allHistoryItems.length
        ? ""
        : "No history yet.";
    }

    if (!state.allHistoryItems.length) {
      return;
    }

    for (const item of state.allHistoryItems) {
      elements.historyList.appendChild(createHistoryItemNode(item));
    }
  }


  // Refill form controls using one history request payload.
  function hydrateFormFromRequest(request) {
    if (!request || typeof request !== "object") {
      return;
    }
    if (elements.text && typeof request.text === "string") {
      elements.text.value = request.text;
    }
    if (elements.modelFamily && request.model_family) {
      elements.modelFamily.value = String(request.model_family).toLowerCase();
      refreshRunOptions();
      syncCustomDropdownSelection(customDropdownControllers.model_family, elements.modelFamily.value);
    }
    if (elements.contentType && request.content_type) {
      elements.contentType.value = String(request.content_type).toLowerCase();
      syncCustomDropdownSelection(customDropdownControllers.content_type, elements.contentType.value);
    }
    if (elements.runId && request.run_id) {
      elements.runId.value = String(request.run_id);
    }
    if (elements.runDropdownValue && request.run_id) {
      elements.runDropdownValue.textContent = String(request.run_id);
    }
    if (elements.topK) {
      elements.topK.value = Number.isFinite(Number(request.top_k)) ? String(request.top_k) : "5";
    }
    if (elements.returnExplanation && typeof request.return_explanation === "boolean") {
      elements.returnExplanation.checked = request.return_explanation;
    }
  }


  // Apply one history item to form and result view.
  function applyHistoryItem(item) {
    if (!item || typeof item !== "object") {
      return;
    }
    hydrateFormFromRequest(item.request ?? {});
    syncRunDropdownSelection(item.request?.run_id ?? "", item.request?.model_family ?? "");
    if (item.meta?.status === "error") {
      renderError(item.error ?? { message: "Failed request from history." });
    } else {
      renderResult(item.response ?? {});
    }
    activateResultTab();
  }


  // Pick a default run_id for one model family.
  function getDefaultRunIdForFamily(modelFamily) {
    const normalized = String(modelFamily ?? "").toLowerCase();
    const familyRuns = state.allRuns.filter((item) => item.model_family === normalized);
    if (!familyRuns.length) {
      return null;
    }
    const best = familyRuns.find((item) => item.is_best);
    return (best ?? familyRuns[0]).run_id;
  }


  // Set result JSON panel and copy button state.
  function setResultJsonPayload(payload) {
    const text = typeof payload === "string" ? payload : JSON.stringify(payload, null, 2);
    state.latestResultJson = text;
    if (elements.resultJson) {
      elements.resultJson.hidden = false;
      elements.resultJson.textContent = text;
    }
    if (elements.copyJsonButton) {
      elements.copyJsonButton.disabled = !text;
    }
  }


  // Copy latest JSON payload to clipboard.
  async function handleCopyJson() {
    const content = state.latestResultJson || "";
    if (!content) {
      setCopyState("No JSON to copy.");
      return;
    }
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(content);
      } else {
        const fallback = document.createElement("textarea");
        fallback.value = content;
        fallback.style.position = "fixed";
        fallback.style.opacity = "0";
        document.body.appendChild(fallback);
        fallback.select();
        document.execCommand("copy");
        fallback.remove();
      }
      setCopyState("Copied.");
    } catch {
      setCopyState("Copy failed.");
    }
  }


  // Load the available runs into the UI.
  function loadRunOptions(runs) {
    state.allRuns = Array.isArray(runs) ? runs : [];
    refreshRunOptions();
    if (state.allHistoryItems.length) {
      renderHistoryList(state.allHistoryItems);
    }
  }


  // Display model configuration information for the selected run.
  function displayModelInfo(runId) {
    const modelInfoPanel = document.getElementById("model-info-panel");
    const subtitleEl = document.getElementById("model-info-subtitle");
    if (!modelInfoPanel) {
      return;
    }

    if (!runId) {
      modelInfoPanel.hidden = true;
      if (subtitleEl) subtitleEl.textContent = "";
      updateRunSelectBestTag("", elements.modelFamily?.value ?? "");
      return;
    }

    const selectedRun = state.allRuns.find((run) => run.run_id === runId);
    if (!selectedRun) {
      modelInfoPanel.hidden = true;
      if (subtitleEl) subtitleEl.textContent = "";
      updateRunSelectBestTag("", elements.modelFamily?.value ?? "");
      return;
    }

    updateRunSelectBestTag(selectedRun.run_id, selectedRun.model_family);

    
    // Helper to format params object into readable lines: key = value
    function formatParams(params, indent = "") {
      if (params === null || params === undefined) return "-";
      if (typeof params === "string") return params || "-";
      if (typeof params === "number" || typeof params === "boolean") return String(params);
      if (Array.isArray(params)) {
        if (!params.length) return "-";
        return params
          .map((item, index) => {
            const value = typeof item === "object" && item !== null ? formatParams(item, `${indent}  `) : String(item);
            return `${indent}${index + 1}. ${value}`;
          })
          .join("\n");
      }
      if (typeof params === "object") {
        try {
          const pairs = [];
          for (const [key, value] of Object.entries(params)) {
            if (value && typeof value === "object") {
              pairs.push(`${indent}${key}:`);
              pairs.push(formatParams(value, `${indent}  `));
            } else {
              pairs.push(`${indent}${key} = ${value}`);
            }
          }
          return pairs.length ? pairs.join("\n") : "-";
        } catch {
          return JSON.stringify(params, null, 2);
        }
      }
      return String(params);
    }

    const modelNameEl = document.getElementById("model-info-name");
    if (modelNameEl) {
      modelNameEl.textContent = selectedRun.model_name || "-";
    }

    const featureSetEl = document.getElementById("model-info-feature-set");
    if (featureSetEl) {
      featureSetEl.textContent = selectedRun.feature_set || "-";
    }

    const textVariantEl = document.getElementById("model-info-text-variant");
    if (textVariantEl) {
      textVariantEl.textContent = selectedRun.text_variant || "-";
    }

    if (subtitleEl) {
      subtitleEl.textContent = selectedRun.model_family === "ml"
        ? "Classical ML model details"
        : "Transformer / deep learning details";
      modelInfoPanel.dataset.family = selectedRun.model_family || "";
    }

    const toggleBtn = document.getElementById("model-info-toggle-btn");
    const globalDefault = Object.prototype.hasOwnProperty.call(state.modelInfoExpanded, '__default')
      ? Boolean(state.modelInfoExpanded['__default'])
      : false;
    const expandedForRun = runId && Object.prototype.hasOwnProperty.call(state.modelInfoExpanded, runId)
      ? Boolean(state.modelInfoExpanded[runId])
      : globalDefault;

    if (expandedForRun) {
      modelInfoPanel.classList.add("expanded");
    } else {
      modelInfoPanel.classList.remove("expanded");
    }
    const isExpanded = expandedForRun;

    const paramsDtEl = document.getElementById("model-info-params-dt");
    const paramsDdEl = document.getElementById("model-info-params");
    if (paramsDtEl) paramsDtEl.hidden = !isExpanded;
    if (paramsDdEl) {
      paramsDdEl.hidden = !isExpanded;
      paramsDdEl.textContent = formatParams(selectedRun.params);
    }

    const thresholdDtEl = document.getElementById("model-info-threshold-dt");
    const thresholdDdEl = document.getElementById("model-info-threshold");
    if (thresholdDtEl) thresholdDtEl.hidden = !isExpanded;
    if (thresholdDdEl) {
      thresholdDdEl.hidden = !isExpanded;
      thresholdDdEl.textContent = selectedRun.threshold === null || selectedRun.threshold === undefined
        ? "-"
        : String(selectedRun.threshold);
    }


    // Helper function to format metric values to 4 decimal places
    function formatMetric(value) {
      if (value === null || value === undefined) return "-";
      const num = parseFloat(value);
      return isNaN(num) ? "-" : (num * 100).toFixed(4) + "%";
    }

    const metricsRow = document.getElementById("model-info-metrics-row");
    const metricMap = [
      { id: "model-info-f1-macro", value: selectedRun.val_f1_macro },
      { id: "model-info-precision-macro", value: selectedRun.val_precision_macro },
      { id: "model-info-recall-macro", value: selectedRun.val_recall_macro },
      { id: "model-info-accuracy", value: selectedRun.val_accuracy },
      { id: "model-info-f1-fake", value: selectedRun.val_f1_fake },
    ];

    let anyMetricVisible = false;
    metricMap.forEach((m) => {
      const el = document.getElementById(m.id);
      if (!el) return;
      if (m.value === null || m.value === undefined) {
        el.textContent = "-";
        el.closest('.metric-item')?.classList.add('hidden');
      } else {
        el.textContent = formatMetric(m.value);
        el.closest('.metric-item')?.classList.remove('hidden');
        anyMetricVisible = true;
      }
    });
    if (metricsRow) {
      metricsRow.hidden = !anyMetricVisible;
    }
    
    const featureSetDtEl = document.getElementById("model-info-feature-set-dt");
    const featureSetDdEl = document.getElementById("model-info-feature-set");
    const textVariantDtEl = document.getElementById("model-info-text-variant-dt");
    const textVariantDdEl = document.getElementById("model-info-text-variant");
    const hasFeatureSet = Boolean(selectedRun.feature_set);
    const hasTextVariant = Boolean(selectedRun.text_variant);
    if (featureSetDtEl) featureSetDtEl.hidden = !isExpanded || !hasFeatureSet;
    if (featureSetDdEl) featureSetDdEl.hidden = !isExpanded || !hasFeatureSet;
    if (textVariantDtEl) textVariantDtEl.hidden = !isExpanded || !hasTextVariant;
    if (textVariantDdEl) textVariantDdEl.hidden = !isExpanded || !hasTextVariant;

    if (toggleBtn && !toggleBtn.dataset.bound) {
      toggleBtn.addEventListener("click", () => {
        const currentRun = String(elements.runId?.value ?? runId ?? "").trim();
        const newExpanded = !modelInfoPanel.classList.contains("expanded");
        modelInfoPanel.classList.toggle("expanded", newExpanded);
        if (currentRun) {
          state.modelInfoExpanded[currentRun] = newExpanded;
        }
        state.modelInfoExpanded['__default'] = newExpanded;
        displayModelInfo(currentRun);
      });
      toggleBtn.dataset.bound = "1";
    }

    if (toggleBtn) {
      toggleBtn.textContent = isExpanded ? "Hide" : "Show more";
    }

    modelInfoPanel.hidden = false;
  }


  // Sort runs by validation metrics: best run first, then higher val_f1_macro and val_f1_fake.
  function sortRunsByMetrics(runs) {
    const metricValue = (run, key) => {
      const value = Number(run?.[key]);
      return Number.isFinite(value) ? value : Number.NEGATIVE_INFINITY;
    };

    return [...runs].sort((a, b) => {
      if (a.is_best && !b.is_best) return -1;
      if (!a.is_best && b.is_best) return 1;

      const f1MacroA = metricValue(a, "val_f1_macro");
      const f1MacroB = metricValue(b, "val_f1_macro");
      if (f1MacroA !== f1MacroB) {
        return f1MacroB - f1MacroA;
      }

      const f1FakeA = metricValue(a, "val_f1_fake");
      const f1FakeB = metricValue(b, "val_f1_fake");
      if (f1FakeA !== f1FakeB) {
        return f1FakeB - f1FakeA;
      }

      const thresholdA = metricValue(a, "threshold");
      const thresholdB = metricValue(b, "threshold");
      if (thresholdA !== thresholdB) {
        return thresholdB - thresholdA;
      }

      return String(a.run_id).localeCompare(String(b.run_id));
    });
  }


  // Refresh the run options in the dropdown based on the selected model family and available runs.
  function refreshRunOptions() {
    const family = String(elements.modelFamily?.value ?? "ml").toLowerCase();
    let filtered = state.allRuns.filter((item) => item.model_family === family);
    const previousValue = elements.runId?.value ?? "";
    if (!elements.runId || !elements.runDropdownToggle) {
      return;
    }

    if (!filtered.length) {
      elements.runDropdownToggle.disabled = true;
      elements.runDropdownToggle.classList.add("is-disabled");
      elements.runDropdownValue.textContent = "No run available";
      if (elements.runStatus) {
        elements.runStatus.textContent = `No ${family.toUpperCase()} run available.`;
      }
      renderRunDropdownOptions([], "");
      syncRunDropdownSelection("", family);
      updateRunSelectBestTag("", family);
      displayModelInfo("");
      return;
    }

    filtered = sortRunsByMetrics(filtered);

    elements.runDropdownToggle.disabled = false;
    elements.runDropdownToggle.classList.remove("is-disabled");
    renderRunDropdownOptions(filtered, previousValue);

    const canKeep = filtered.some((item) => item.run_id === previousValue);
    let selectedRunId;
    if (canKeep) {
      selectedRunId = previousValue;
    } else {
      const bestRun = filtered.find((item) => item.is_best);
      selectedRunId = bestRun ? bestRun.run_id : filtered[0].run_id;
    }
    if (elements.runStatus) {
      elements.runStatus.textContent = `${filtered.length} run(s) loaded for ${family.toUpperCase()}.`;
    }
    syncRunDropdownSelection(selectedRunId, family);
    updateRunSelectBestTag(selectedRunId, family);
    displayModelInfo(selectedRunId);
  }


  // Set the loading state for the run options, showing a message while runs are being fetched from the backend.
  function setRunLoadingState(message) {
    if (elements.runDropdownToggle) {
      elements.runDropdownToggle.disabled = true;
      elements.runDropdownToggle.classList.add("is-disabled");
      if (elements.runDropdownValue) {
        elements.runDropdownValue.textContent = message;
      }
    }
    if (elements.runStatus) {
      elements.runStatus.textContent = message;
    }
  }


  // Get the value of a specific form field, applying necessary normalization and type conversion.
  function getPayload() {
    const textRaw = String(elements.text?.value ?? "");
    const normalizedText = normalizeInputText(textRaw);
    const modelFamily = String(elements.modelFamily?.value ?? "").toLowerCase().trim();
    const contentType = String(elements.contentType?.value ?? "").toLowerCase().trim();
    const runId = String(elements.runId?.value ?? "").trim();
    const topKRaw = String(elements.topK?.value ?? "").trim();

    return {
      text: normalizedText,
      model_family: modelFamily,
      content_type: contentType,
      run_id: runId,
      top_k: topKRaw ? Number(topKRaw) : null,
      return_explanation: Boolean(elements.returnExplanation?.checked),
      _rawText: normalizedText,
    };
  }


  // Validate the prediction payload, checking for required fields and ensuring values are within expected ranges or formats.
  function validatePayload(payload) {
    const errors = {};
    const meaningfulLen = payload._rawText.replace(/\s+/g, "").length;
    if (!payload.text || meaningfulLen <= 20) {
      errors.text = "Text must contain meaningful content with more than 20 non-whitespace characters.";
    }
    if (!VALID_FAMILIES.has(payload.model_family)) {
      errors.model_family = "Model family must be ML or DL.";
    }
    if (!VALID_CONTENT_TYPES.has(payload.content_type)) {
      errors.content_type = "Content type must be News or Social.";
    }
    if (!payload.run_id) {
      errors.run_id = "Run ID is required.";
    } else {
      const allowedRuns = state.allRuns
        .filter((item) => item.model_family === payload.model_family)
        .map((item) => item.run_id);
      if (allowedRuns.length && !allowedRuns.includes(payload.run_id)) {
        errors.run_id = "Selected run_id does not belong to current model family.";
      }
    }
    if (!Number.isInteger(payload.top_k) || payload.top_k < 1 || payload.top_k > 30) {
      errors.top_k = "Top K must be an integer in range [1,30].";
    }
    return errors;
  }


  // Render the prediction result in the UI, showing a success message and displaying the result JSON.
  function renderResult(result) {
    if (elements.resultNote) {
      elements.resultNote.textContent = "Prediction completed successfully.";
    }
    if (elements.resultErrorBox) {
      elements.resultErrorBox.hidden = true;
      elements.resultErrorBox.textContent = "";
    }

    if (elements.resultSummary) {
      elements.resultSummary.hidden = false;
    }

    const label = resolveLabel(result);
    if (elements.resultBadge) {
      elements.resultBadge.className = label.className;
      elements.resultBadge.textContent = label.text;
    }
    if (elements.resultConfidence) {
      elements.resultConfidence.textContent = `Confidence: ${formatPercent(result?.confidence)}`;
    }

    renderKeyValueGrid(elements.resultMetricGrid, [
      { key: "Threshold", value: String(result?.threshold_used ?? "-") },
      { key: "Raw Score", value: String(result?.raw_score ?? "-") },
      { key: "Processing Time", value: formatMs(result?.processing_time_ms) },
      { key: "Score Method", value: String(result?.score_method ?? "-") },
    ]);

    renderKeyValueGrid(elements.resultModelGrid, [
      { key: "Model Type (Model Family)", value: String(result?.model_family ?? "-").toUpperCase() },
      {
        key: "Model Selected (Run ID)",
        value: String(result?.run_id ?? "-"),
        ...(function () {
          const rank = getRunRank(result?.run_id, result?.model_family);
          if (!rank || rank > 3) return {};
          return { badgeText: rank === 1 ? "1st" : rank === 2 ? "2nd" : "3rd", badgeClass: `run-rank-${rank}` };
        })(),
      },
      { key: "Model Name", value: String(result?.model_name ?? "-") },
      { key: "Feature Set", value: String(result?.feature_set ?? "-") },
    ]);

    if (elements.resultWarnings) {
      const warnings = Array.isArray(result?.warnings) ? result.warnings : [];
      elements.resultWarnings.innerHTML = "";
      if (!warnings.length) {
        elements.resultWarnings.hidden = true;
      } else {
        elements.resultWarnings.hidden = false;
        for (const warning of warnings) {
          const item = document.createElement("li");
          item.textContent = String(warning);
          elements.resultWarnings.appendChild(item);
        }
      }
    }

    renderExplanation(result);
    setResultJsonPayload(result);
    setCopyState("");
  }


  // Render an error message in the UI, showing the provided message in the result note area.
  function renderError(message) {
    const asObject = typeof message === "object" && message !== null ? message : null;
    const errorCode = asObject?.errorCode ? String(asObject.errorCode) : "";
    const detail = asObject?.message ? String(asObject.message) : String(message ?? "Unknown error.");
    const combinedMessage = errorCode ? `[${errorCode}] ${detail}` : detail;

    if (elements.resultNote) {
      elements.resultNote.textContent = "Prediction failed.";
    }
    if (elements.resultSummary) {
      elements.resultSummary.hidden = true;
    }
    if (elements.resultErrorBox) {
      elements.resultErrorBox.hidden = false;
      elements.resultErrorBox.textContent = combinedMessage;
    }
    setResultJsonPayload(
      asObject
        ? {
            type: asObject.type ?? "api_error",
            status: asObject.status ?? null,
            error_code: asObject.errorCode ?? null,
            detail: asObject.message ?? detail,
            raw: asObject.raw ?? null,
          }
        : { detail: combinedMessage },
    );
    resetExplanation();
    setCopyState("");
  }

  if (elements.copyJsonButton) {
    elements.copyJsonButton.addEventListener("click", () => {
      handleCopyJson();
    });
  }

  initCustomDropdownController({
    key: "content_type",
    container: elements.contentTypeDropdown,
    toggle: elements.contentTypeDropdownToggle,
    value: elements.contentTypeDropdownValue,
    menu: elements.contentTypeDropdownMenu,
    select: elements.contentType,
  });
  initCustomDropdownController({
    key: "model_family",
    container: elements.modelFamilyDropdown,
    toggle: elements.modelFamilyDropdownToggle,
    value: elements.modelFamilyDropdownValue,
    menu: elements.modelFamilyDropdownMenu,
    select: elements.modelFamily,
  });
  bindCustomDropdownOutsideClick();
  initRunDropdown();
  initTopKSpinnerOnlyInput();

  syncCustomDropdownSelection(customDropdownControllers.content_type, elements.contentType?.value ?? "news");
  syncCustomDropdownSelection(customDropdownControllers.model_family, elements.modelFamily?.value ?? "ml");

  return {
    elements,
    setStatus,
    setFormError,
    setFieldError,
    clearFieldErrors,
    setLoading,
    loadRunOptions,
    refreshRunOptions,
    setRunLoadingState,
    displayModelInfo,
    getPayload,
    validatePayload,
    renderResult,
    renderError,
    renderHistoryList,
    applyHistoryItem,
    getDefaultRunIdForFamily,
  };
}
