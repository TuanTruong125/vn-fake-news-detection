const DEFAULT_API_BASE = "";

export function createApiClient(options = {}) {
  const baseUrl = options.baseUrl ?? DEFAULT_API_BASE;

  async function request(path, init = {}) {
    const response = await fetch(`${baseUrl}${path}`, init);
    const contentType = response.headers.get("content-type") ?? "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : await response.text();

    if (!response.ok) {
      throw new Error(
        typeof payload === "string"
          ? payload
          : payload?.detail ?? "Request failed.",
      );
    }
    return payload;
  }

  async function predict(payload) {
    return request("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  }

  return {
    request,
    predict,
  };
}
