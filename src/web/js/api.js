const DEFAULT_API_BASE = resolveDefaultApiBase();
const DEFAULT_TIMEOUT_MS = 30000;


// Heuristic to determine default API base URL based on current window location.
function resolveDefaultApiBase() {
  const hasWindow = typeof window !== "undefined" && Boolean(window.location);
  const host = hasWindow ? window.location.hostname : "127.0.0.1";
  const port = hasWindow ? window.location.port : "";
  if (port === "5500" || port === "5501" || port === "5502" || port === "5503") {
    return `http://${host}:8000`;
  }
  return "";
}


// FNV-1a 32-bit hash function for generating text fingerprints.
function fnv1a32(input) {
  let hash = 0x811c9dc5;
  for (let i = 0; i < input.length; i += 1) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}


// Factory function to create an API client with configurable base URL and timeout.
export function createApiClient(options = {}) {
  const baseUrl = options.baseUrl ?? DEFAULT_API_BASE;
  const timeoutMs = Number.isFinite(options.timeoutMs) ? options.timeoutMs : DEFAULT_TIMEOUT_MS;

  // Fetch and filter runs based on selected model family, then update the run ID dropdown.
  async function request(path, init = {}) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    let response;
    try {
      response = await fetch(`${baseUrl}${path}`, {
        ...init,
        signal: controller.signal,
      });
    } catch (error) {
      clearTimeout(timer);
      if (error?.name === "AbortError") {
        throw {
          type: "network_error",
          status: null,
          errorCode: "REQUEST_TIMEOUT",
          message: "Request timed out. Please try again.",
        };
      }
      throw {
        type: "network_error",
        status: null,
        errorCode: "NETWORK_ERROR",
        message: "Cannot reach backend service.",
      };
    } finally {
      clearTimeout(timer);
    }

    const contentType = response.headers.get("content-type") ?? "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : await response.text();

    if (!response.ok) {
      let message =
        typeof payload === "string"
          ? payload
          : payload?.detail ?? "Request failed.";
      if (typeof message === "string" && message.trim().startsWith("<!DOCTYPE html>")) {
        if (path === "/runs") {
          message = "Run list endpoint is unavailable. Please ensure backend API is running on port 8000.";
        } else {
          message = "Backend returned a non-JSON error response.";
        }
      }
      throw {
        type: "api_error",
        status: response.status,
        errorCode: typeof payload === "object" && payload ? payload.error_code ?? null : null,
        detail: typeof payload === "object" && payload ? payload.detail ?? null : null,
        message,
        raw: payload,
      };
    }
    return payload;
  }


  // Prepare payload for prediction request, including text normalization and input validation.
  async function predict(payload) {
    const text = String(payload?.text ?? "");
    const newlineCount = (text.match(/\n/g) ?? []).length;
    const nonWhitespaceLength = text.replace(/\s+/g, "").length;
    const fingerprint = `fnv1a32:${fnv1a32(text)}|len:${text.length}|nw:${nonWhitespaceLength}|nl:${newlineCount}`;
    return request("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Client-Text-Fingerprint": fingerprint,
      },
      body: JSON.stringify(payload),
    });
  }

  
  // Fetch and filter runs based on selected model family, then update the run ID dropdown.
  async function fetchRuns() {
    const payload = await request("/runs", {
      method: "GET",
    });
    const runs = Array.isArray(payload?.runs) ? payload.runs : [];
    return runs
      .filter((item) => item && typeof item.run_id === "string" && typeof item.model_family === "string")
      .map((item) => ({
        run_id: String(item.run_id).trim(),
        model_family: String(item.model_family).toLowerCase(),
        is_best: Boolean(item.is_best),
        // preserve optional metadata fields if present
        model_name: item.model_name ?? null,
        feature_set: item.feature_set ?? null,
        text_variant: item.text_variant ?? null,
        params: item.params ?? null,
        threshold: item.threshold ?? null,
      }));
  }

  return {
    request,
    predict,
    fetchRuns,
  };
}
