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

      // Remove active class from all buttons and hide all content
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

      // Add active class to clicked button and show corresponding content
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
    contentType: document.getElementById("content_type"),
    runId: document.getElementById("run_id"),
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
  };


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
        return elements.modelFamily;
      case "content_type":
        return elements.contentType;
      case "run_id":
        return elements.runId;
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
      value.className = "result-kv-value";
      value.textContent = row.value;
      item.appendChild(key);
      item.appendChild(value);
      target.appendChild(item);
    }
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
  }


  // Refresh the run options in the dropdown based on the selected model family and available runs.
  function refreshRunOptions() {
    const family = String(elements.modelFamily?.value ?? "ml").toLowerCase();
    const filtered = state.allRuns.filter((item) => item.model_family === family);
    const previousValue = elements.runId?.value ?? "";
    if (!elements.runId) {
      return;
    }

    elements.runId.innerHTML = "";
    if (!filtered.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No run available";
      elements.runId.appendChild(option);
      if (elements.runStatus) {
        elements.runStatus.textContent = `No ${family.toUpperCase()} run available from backend.`;
      }
      return;
    }

    for (const run of filtered) {
      const option = document.createElement("option");
      option.value = run.run_id;
      option.textContent = run.is_best ? `${run.run_id} (BEST)` : run.run_id;
      elements.runId.appendChild(option);
    }

    const canKeep = filtered.some((item) => item.run_id === previousValue);
    elements.runId.value = canKeep ? previousValue : filtered[0].run_id;
    if (elements.runStatus) {
      elements.runStatus.textContent = `${filtered.length} run(s) loaded for ${family.toUpperCase()}.`;
    }
  }


  // Set the loading state for the run options, showing a message while runs are being fetched from the backend.
  function setRunLoadingState(message) {
    if (elements.runId) {
      elements.runId.innerHTML = `<option value="">${message}</option>`;
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
      { key: "Model Selected (Run ID)", value: String(result?.run_id ?? "-") },
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
    setCopyState("");
  }

  if (elements.copyJsonButton) {
    elements.copyJsonButton.addEventListener("click", () => {
      handleCopyJson();
    });
  }

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
    getPayload,
    validatePayload,
    renderResult,
    renderError,
  };
}
