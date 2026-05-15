// Fixture: Complete web app DOM structure for browser-based test harnesses
export const WEB_APP_HTML = `
  <main class="main-layout" id="app-main">
    <section class="predict-input-section" id="predict-panel">
      <form id="predict-form" novalidate>
        <div class="form-error" id="form-error" role="alert" aria-live="polite" hidden></div>

        <textarea id="text" name="text" rows="4" required aria-describedby="text-error"></textarea>
        <p class="field-error" id="text-error" hidden></p>

        <div id="content-type-dropdown" class="run-dropdown">
          <button id="content-type-dropdown-toggle" class="run-dropdown-toggle" type="button" aria-haspopup="listbox" aria-expanded="false" aria-controls="content-type-dropdown-menu">
            <span id="content-type-dropdown-value" class="run-dropdown-value">News</span>
            <span class="run-dropdown-caret" aria-hidden="true"></span>
          </button>
          <div id="content-type-dropdown-menu" class="run-dropdown-menu" role="listbox" hidden></div>
        </div>
        <select id="content_type" name="content_type" required hidden aria-hidden="true" tabindex="-1">
          <option value="news">News</option>
          <option value="social">Social</option>
        </select>
        <p class="field-error" id="content-type-error" hidden></p>

        <input id="top_k" name="top_k" type="number" min="1" max="30" step="1" value="5" aria-describedby="top-k-error">
        <p class="field-error" id="top-k-error" hidden></p>

        <div id="model-family-dropdown" class="run-dropdown">
          <button id="model-family-dropdown-toggle" class="run-dropdown-toggle" type="button" aria-haspopup="listbox" aria-expanded="false" aria-controls="model-family-dropdown-menu">
            <span id="model-family-dropdown-value" class="run-dropdown-value">ML</span>
            <span class="run-dropdown-caret" aria-hidden="true"></span>
          </button>
          <div id="model-family-dropdown-menu" class="run-dropdown-menu" role="listbox" hidden></div>
        </div>
        <select id="model_family" name="model_family" required hidden aria-hidden="true" tabindex="-1">
          <option value="ml">ML</option>
          <option value="dl">DL</option>
        </select>
        <p class="field-error" id="model-family-error" hidden></p>

        <div id="run-dropdown" class="run-dropdown">
          <button id="run-dropdown-toggle" class="run-dropdown-toggle" type="button" aria-haspopup="listbox" aria-expanded="false" aria-controls="run-dropdown-menu">
            <span id="run-dropdown-value" class="run-dropdown-value">Loading runs...</span>
            <span class="run-dropdown-caret" aria-hidden="true"></span>
          </button>
          <div id="run-dropdown-menu" class="run-dropdown-menu" role="listbox" hidden></div>
        </div>
        <select id="run_id" name="run_id" required hidden aria-hidden="true" tabindex="-1">
          <option value="">Loading runs...</option>
        </select>
        <p id="run-status" aria-live="polite">Loading runs...</p>
        <p class="field-error" id="run-id-error" hidden></p>

        <div id="model-info-panel" class="model-info-panel" hidden>
          <div class="model-info-header">
            <h3 class="model-info-title">Model Configuration</h3>
            <div class="model-info-header-actions">
              <p class="model-info-subtitle" id="model-info-subtitle"></p>
              <button id="model-info-toggle-btn" class="model-info-toggle" type="button">Show more</button>
            </div>
          </div>
          <dl class="model-info-list">
            <dt id="model-info-name-dt">Model</dt>
            <dd id="model-info-name">-</dd>
            <dt id="model-info-feature-set-dt">Feature Set</dt>
            <dd id="model-info-feature-set">-</dd>
            <dt id="model-info-text-variant-dt">Text Variant</dt>
            <dd id="model-info-text-variant">-</dd>
            <dt id="model-info-params-dt" hidden>Parameters</dt>
            <dd id="model-info-params" hidden>-</dd>
            <dt id="model-info-threshold-dt" hidden>Threshold</dt>
            <dd id="model-info-threshold" hidden>-</dd>
            <dt class="model-info-metrics-divider">Validation Metrics</dt>
            <dd class="model-info-metrics-row" id="model-info-metrics-row">
              <div class="metric-item"><div class="metric-label">F1 (Macro)</div><div id="model-info-f1-macro" class="metric-value">-</div></div>
              <div class="metric-item"><div class="metric-label">Precision (Macro)</div><div id="model-info-precision-macro" class="metric-value">-</div></div>
              <div class="metric-item"><div class="metric-label">Recall (Macro)</div><div id="model-info-recall-macro" class="metric-value">-</div></div>
              <div class="metric-item"><div class="metric-label">Accuracy</div><div id="model-info-accuracy" class="metric-value">-</div></div>
              <div class="metric-item"><div class="metric-label">F1 (Fake)</div><div id="model-info-f1-fake" class="metric-value">-</div></div>
            </dd>
          </dl>
        </div>

        <label for="return_explanation"><input id="return_explanation" name="return_explanation" type="checkbox" checked> Return Explanation</label>
        <button id="predict-button" class="predict-btn" type="submit"><span class="btn-label">Predict</span></button>
      </form>
    </section>

    <section class="result-section" id="result-panel">
      <div class="tabs-header">
        <button class="tab-button active" data-tab="result-tab" aria-selected="true">Result</button>
        <button class="tab-button" data-tab="history-tab" aria-selected="false">History</button>
      </div>

      <div class="tab-content active" id="result-tab" role="tabpanel" aria-labelledby="result-tab">
        <p class="panel-note" id="result-note">Submit the form to check the prediction results.</p>
        <div class="result-summary" id="result-summary" hidden>
          <div class="result-headline"><span class="result-badge" id="result-badge">-</span><strong id="result-confidence">-</strong></div>
          <div class="result-metric-grid" id="result-metric-grid"></div>
          <div class="result-model-grid" id="result-model-grid"></div>
          <ul class="result-warnings" id="result-warnings" hidden></ul>
        </div>
        <div class="result-error-box" id="result-error-box" hidden></div>
        <div class="result-json-toolbar"><button type="button" id="copy-json-btn" class="copy-json-btn" disabled>Copy JSON</button><span class="copy-json-state" id="copy-json-state" aria-live="polite"></span></div>
        <pre class="result-json" id="result-json" hidden></pre>

        <section id="explain-panel" class="explain-panel" hidden>
          <div class="explain-head"><h3>Explanation</h3><p id="explain-status" class="explain-status"></p></div>
          <div id="explain-empty" class="explain-empty" hidden></div>
          <div id="explain-content" hidden>
            <div class="explain-columns">
              <div class="explain-col fake"><h4>Towards FAKE</h4><div id="explain-fake-list" class="explain-list"></div></div>
              <div class="explain-col real"><h4>Towards REAL</h4><div id="explain-real-list" class="explain-list"></div></div>
            </div>
            <div class="explain-decomposition"><h4>Decomposition</h4><div id="explain-decomposition-grid" class="explain-decomposition-grid"></div><p id="explain-decomposition-note" class="explain-decomposition-note"></p></div>
          </div>
        </section>
      </div>

      <div class="tab-content" id="history-tab" role="tabpanel" aria-labelledby="history-tab" hidden>
        <div id="history-widgets" class="history-widgets">
          <div class="history-toolbar"><strong>Prediction History</strong><button type="button" id="history-clear-btn" class="history-clear-btn">Clear</button></div>
          <div id="history-empty" class="history-empty" hidden></div>
        </div>
        <div class="history-placeholder"><div id="history-list" class="history-list"></div></div>
      </div>
    </section>
  </main>

  <div id="test-result"></div>
`;
