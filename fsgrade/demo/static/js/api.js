/* Thin fetch wrapper.
   Every backend error arrives as {error:{code,message,remediation}}, so it is
   surfaced as a typed exception rather than a bare status code. */
(function (global) {
  "use strict";

  class ApiError extends Error {
    constructor(payload, status) {
      const err = (payload && payload.error) || {};
      super(err.message || `Request failed (${status})`);
      this.name = "ApiError";
      this.code = err.code || "http_error";
      this.remediation = err.remediation || "";
      this.status = status;
      this.details = err.details || {};
    }
  }

  let inflight = null; // one prediction at a time; a newer click cancels the older

  async function request(path, options = {}) {
    const res = await fetch(path, options);
    let payload = null;
    if (res.status !== 204) {
      try { payload = await res.json(); } catch (_) { payload = null; }
    }
    if (!res.ok) throw new ApiError(payload, res.status);
    return payload;
  }

  const json = (path, method, body) =>
    request(path, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });

  const Api = {
    ApiError,

    health:  () => request("/api/health"),
    runs:    () => request("/api/runs"),
    species: () => request("/api/species"),
    methods: () => request("/api/methods"),

    createSession: (body) => json("/api/sessions", "POST", body || {}),
    getSession:    (sid) => request(`/api/sessions/${sid}`),
    patchSession:  (sid, body) => json(`/api/sessions/${sid}`, "PATCH", body),

    sample: (sid, group, body) =>
      json(`/api/sessions/${sid}/${group}/sample`, "POST", body || {}),

    upload(sid, group, files, label) {
      const form = new FormData();
      for (const f of files) form.append("files", f, f.name);
      if (label) form.append("label", label);
      return request(`/api/sessions/${sid}/${group}/upload`, { method: "POST", body: form });
    },

    removeImage: (sid, group, id) =>
      request(`/api/sessions/${sid}/${group}/${id}`, { method: "DELETE" }),
    clear:   (sid, group) => json(`/api/sessions/${sid}/${group}/clear`, "POST", {}),
    balance: (sid) => json(`/api/sessions/${sid}/support/balance`, "POST", {}),
    labelQuery: (sid, id, label) =>
      json(`/api/sessions/${sid}/query/${id}`, "PATCH", { label }),

    /* Predictions supersede one another: an examiner dragging the kappa dial
       fires many, and only the last answer is worth rendering. */
    async predict(sid, body) {
      if (inflight) inflight.abort();
      const controller = new AbortController();
      inflight = controller;
      try {
        return await request(`/api/sessions/${sid}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body || {}),
          signal: controller.signal,
        });
      } finally {
        if (inflight === controller) inflight = null;
      }
    },
  };

  global.Api = Api;
})(window);
