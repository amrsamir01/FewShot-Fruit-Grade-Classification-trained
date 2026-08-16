/* Boot, wiring, rendering. No framework, no build step. */
(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  let plane = null;
  let kappaTimer = null;

  /* ------------------------------------------------------------- boot -- */
  async function boot() {
    plane = new DecisionPlane($("#plane"));

    try {
      const [health, species, methods] = await Promise.all([
        Api.health(), Api.species(), Api.methods(),
      ]);
      Store.set({
        health, methods: methods.methods,
        speciesList: species.species || [],
        kappa: methods.kappa || 5,
        kappaCalibrated: methods.kappa,
        kappaSource: methods.kappa_source || "default, not calibrated",
        ready: true,
      });
      if (health.warnings && health.warnings.length) {
        Store.banner(health.warnings[0], "warn");
      }

      const session = await Api.createSession({ species: pickDefaultSpecies(species) });
      Store.setSession(session.session);
      syncControlsFromSession();
    } catch (err) {
      Store.banner(describe(err), "error");
    }
    wire();
  }

  function pickDefaultSpecies(species) {
    const names = (species.species || []).map((s) => s.name);
    return names.includes("mango") ? "mango" : (names[0] || "mango");
  }

  function describe(err) {
    if (err && err.remediation) return `${err.message} — ${err.remediation}`;
    return (err && err.message) || String(err);
  }

  /* ------------------------------------------------------------- wire -- */
  function wire() {
    document.addEventListener("click", onClick);

    $("#species").addEventListener("change", async (e) => {
      await patch({ species: e.target.value.trim() || "mango" });
      plane.reset();
      Store.set({ result: null });
    });

    $("#arm-kshot").addEventListener("change", (e) => patch({ arm_kshot: e.target.value }));
    $("#arm-zeroshot").addEventListener("change", (e) =>
      patch({ arm_zeroshot: e.target.value || null }));

    const kappa = $("#kappa");
    kappa.addEventListener("input", (e) => {
      const v = parseFloat(e.target.value);
      Store.set({ kappa: v });
      clearTimeout(kappaTimer);
      // Re-blend live while dragging, but do not fire a request per pixel.
      kappaTimer = setTimeout(() => { if (Store.state.result) predict(); }, 140);
    });

    setupDrop($("#query-strip"), "query", null);
    document.addEventListener("paste", onPaste);
  }

  async function patch(body) {
    const s = Store.state.session;
    if (!s) return;
    try {
      const res = await Api.patchSession(s.session_id, body);
      Store.setSession(res.session);
    } catch (err) { Store.banner(describe(err), "error"); }
  }

  async function onClick(e) {
    const btn = e.target.closest("[data-action]");
    if (btn) {
      e.preventDefault();
      return act(btn.dataset.action);
    }
    const thumb = e.target.closest(".thumb");
    if (thumb && thumb.dataset.id) {
      const s = Store.state.session;
      try {
        const res = await Api.removeImage(s.session_id, thumb.dataset.group, thumb.dataset.id);
        Store.setSession(res.session);
        if (Store.state.result) predict();
      } catch (err) { Store.banner(describe(err), "error"); }
    }
  }

  async function act(action) {
    const s = Store.state.session;
    if (!s) return;
    Store.busy(true);
    Store.clearBanner();
    try {
      if (action === "sample-support") {
        Store.setSession((await Api.sample(s.session_id, "support", { n_per_class: 5 })).session);
      } else if (action === "sample-query") {
        Store.setSession((await Api.sample(s.session_id, "query", { n_per_class: 4 })).session);
      } else if (action === "clear-support") {
        Store.setSession((await Api.clear(s.session_id, "support")).session);
        plane.reset(); Store.set({ result: null });
      } else if (action === "clear-query") {
        Store.setSession((await Api.clear(s.session_id, "query")).session);
        plane.reset(); Store.set({ result: null });
      } else if (action === "balance") {
        Store.setSession((await Api.balance(s.session_id)).session);
      } else if (action === "predict") {
        await predict();
      }
    } catch (err) {
      Store.banner(describe(err), "error");
    } finally {
      Store.busy(false);
    }
  }

  async function predict() {
    const s = Store.state.session;
    if (!s || !s.query.length) return;
    Store.busy(true);
    try {
      const payload = await Api.predict(s.session_id, { kappa: Store.state.kappa });
      Store.setResult(payload);
    } catch (err) {
      if (err.name === "AbortError") return;      // superseded by a newer click
      Store.banner(describe(err), "error");
    } finally {
      Store.busy(false);
    }
  }

  /* ------------------------------------------------------ drag & drop -- */
  function setupDrop(node, group, label) {
    if (!node) return;
    ["dragenter", "dragover"].forEach((t) =>
      node.addEventListener(t, (e) => { e.preventDefault(); node.classList.add("dragover"); }));
    ["dragleave", "drop"].forEach((t) =>
      node.addEventListener(t, () => node.classList.remove("dragover")));
    node.addEventListener("drop", async (e) => {
      e.preventDefault();
      const files = [...(e.dataTransfer?.files || [])].filter((f) => f.type.startsWith("image/"));
      if (files.length) await upload(group, files, label);
    });
  }

  async function onPaste(e) {
    const files = [...(e.clipboardData?.files || [])].filter((f) => f.type.startsWith("image/"));
    if (files.length) await upload("query", files, null);
  }

  async function upload(group, files, label) {
    const s = Store.state.session;
    if (!s) return;
    Store.busy(true);
    try {
      const res = await Api.upload(s.session_id, group, files, label);
      Store.setSession(res.session);
      if (res.rejected && res.rejected.length) {
        Store.banner(
          res.rejected.map((r) => `${r.filename}: ${r.reason}`).join(" · "), "warn");
      }
    } catch (err) {
      Store.banner(describe(err), "error");
    } finally { Store.busy(false); }
  }

  /* ---------------------------------------------------------- render -- */
  function syncControlsFromSession() {
    const s = Store.state.session;
    if (!s) return;
    $("#species").value = s.species;
    fillSelect($("#arm-kshot"), (m) => m.available, s.arm_kshot);
    fillSelect($("#arm-zeroshot"), (m) => m.available && !usesSupport(m.name),
      s.arm_zeroshot, true);
  }

  const ZERO_SHOT = new Set(["zeroshot_supervised", "clip_text_zeroshot", "chance"]);
  const usesSupport = (name) => !ZERO_SHOT.has(name);

  function fillSelect(select, filter, value, allowNone) {
    if (!select) return;
    const opts = Store.state.methods.filter(filter);
    select.innerHTML = "";
    if (allowNone) select.appendChild(new Option("— none —", ""));
    for (const m of opts) {
      const o = new Option(m.name, m.name);
      select.appendChild(o);
    }
    for (const m of Store.state.methods.filter((x) => !x.available && filter({ ...x, available: true }))) {
      const o = new Option(`${m.name} (${m.reason})`, m.name);
      o.disabled = true;
      select.appendChild(o);
    }
    if (value) select.value = value;
  }

  Store.subscribe(render);

  function render(state) {
    renderStatus(state);
    renderBanner(state);
    if (!state.session) return;
    renderTrays(state);
    renderQueries(state);
    renderKappa(state);
    renderVerdicts(state);
    renderPlane(state);
    renderFooter(state);

    document.querySelectorAll("[data-action]").forEach((b) => { b.disabled = state.busy; });
    $("#k-badge").textContent = `K = ${state.session.k_shot}`;
  }

  function renderStatus(state) {
    const bar = $("#status-bar");
    if (!state.health) { bar.textContent = "starting…"; return; }
    const c = state.health.capabilities;
    bar.innerHTML = "";
    const tag = (text, kind) => bar.appendChild(el("span", `tag ${kind}`, text));
    tag(state.health.device.toUpperCase(), "ok");
    tag(c.dataset ? "dataset" : "no dataset", c.dataset ? "ok" : "off");
    tag(c.clip ? "CLIP" : "CLIP off", c.clip ? "ok" : "off");
    tag(`${c.checkpoint_arms.length} trained`, c.checkpoint_arms.length ? "ok" : "warn");
  }

  function renderBanner(state) {
    const node = $("#banner");
    if (!state.banner) { node.hidden = true; return; }
    node.hidden = false;
    node.className = `banner ${state.banner.kind === "error" ? "error" : ""}`;
    node.textContent = state.banner.text;
  }

  function renderTrays(state) {
    const wrap = $("#trays");
    const s = state.session;
    if (wrap.dataset.rev === String(s.revision)) return;
    wrap.dataset.rev = String(s.revision);
    wrap.innerHTML = "";

    s.classes.forEach((cls, index) => {
      const tray = el("div", `tray ${cls}`);
      const head = el("div", "tray-head");
      head.appendChild(el("span", null, cls));
      const n = s.support.filter((c) => c.label === index).length;
      head.appendChild(el("span", "tray-count", String(n)));
      tray.appendChild(head);

      const thumbs = el("div", "thumbs");
      s.support.filter((c) => c.label === index).forEach((card) => {
        thumbs.appendChild(thumbNode(card, "support"));
      });
      tray.appendChild(thumbs);
      if (!n) tray.appendChild(el("p", "tray-hint", "drop images here"));
      wrap.appendChild(tray);
      setupDrop(tray, "support", cls);
    });
  }

  function thumbNode(card, group, extra) {
    const node = el("button", `thumb ${extra || ""}`);
    node.type = "button";
    node.dataset.id = card.image_id;
    node.dataset.group = group;
    node.title = `${card.filename} — click to remove`;
    const img = el("img");
    img.src = card.thumb;
    img.alt = card.filename;
    node.appendChild(img);
    return node;
  }

  function renderQueries(state) {
    const strip = $("#query-strip");
    const s = state.session;
    const result = state.result;
    const primary = Store.kShotVerdict() || Store.primaryVerdict();

    strip.innerHTML = "";
    s.query.forEach((card, i) => {
      let extra = "";
      if (primary && primary.predictions.length > i) {
        const pred = primary.predictions[i];
        extra = pred === 1 ? "pred-rotten" : "pred-fresh";
        if (card.label >= 0 && card.label !== pred) extra += " wrong";
      }
      strip.appendChild(thumbNode(card, "query", extra));
    });
    if (!s.query.length) strip.appendChild(el("p", "tray-hint", "drop or paste query images"));
    $("#query-count").textContent = String(s.query.length);
  }

  function renderKappa(state) {
    const block = $("#kappa-block");
    const usesKappa = state.session.arm_kshot === "sap";
    block.hidden = !usesKappa;
    if (!usesKappa) return;

    $("#kappa").value = state.kappa;
    $("#kappa-value").textContent = state.kappa.toFixed(1);

    const k = state.session.k_shot;
    const alpha = k + state.kappa > 0 ? k / (k + state.kappa) : 0;
    $("#alpha-readout").textContent = `α = ${alpha.toFixed(2)}`;

    const foot = $("#kappa-source").parentElement;
    const calibrated = state.kappaCalibrated;
    if (calibrated != null && Math.abs(state.kappa - calibrated) > 1e-6) {
      $("#kappa-source").textContent = `exploring — κ ≠ calibrated (${calibrated})`;
      foot.classList.add("exploring");
    } else {
      $("#kappa-source").textContent = state.kappaSource;
      foot.classList.remove("exploring");
    }
  }

  function renderVerdicts(state) {
    const box = $("#verdicts");
    box.innerHTML = "";
    if (!state.result) return;

    const kshot = Store.kShotVerdict();
    const zero = Store.zeroShotVerdict();

    state.result.verdicts.forEach((v) => {
      const card = el("div", `verdict ${v.error ? "err" : ""} ${v === kshot ? "primary" : ""}`);
      const head = el("div", "verdict-head");
      const left = el("div");
      left.appendChild(el("div", "verdict-name", v.arm));
      left.appendChild(el("div", "verdict-sub",
        v.uses_support ? `${v.k_shot}-shot · ${(v.seconds * 1000).toFixed(0)} ms`
                       : `0 target labels · ${(v.seconds * 1000).toFixed(0)} ms`));
      head.appendChild(left);

      if (v.error) {
        card.appendChild(head);
        card.appendChild(el("p", "verdict-err-msg", v.error));
        box.appendChild(card);
        return;
      }

      const acc = v.accuracy;
      head.appendChild(el("div", "verdict-acc", acc == null ? "—" : `${(acc * 100).toFixed(1)}%`));
      card.appendChild(head);

      const bar = el("div", "bar");
      const fill = el("i");
      fill.style.width = `${(acc == null ? 0 : acc) * 100}%`;
      bar.appendChild(fill);
      card.appendChild(bar);
      box.appendChild(card);
    });

    // The comparison the thesis is actually about.
    if (kshot && zero && kshot.accuracy != null && zero.accuracy != null) {
      const d = kshot.accuracy - zero.accuracy;
      const kind = Math.abs(d) < 1e-9 ? "flat" : (d > 0 ? "up" : "down");
      const row = el("div", "verdict");
      const head = el("div", "verdict-head");
      head.appendChild(el("div", "verdict-name", "what the support set bought"));
      head.appendChild(el("span", `delta ${kind}`,
        `${d >= 0 ? "+" : ""}${(d * 100).toFixed(1)} pts`));
      row.appendChild(head);
      row.appendChild(el("div", "verdict-sub",
        `${kshot.arm} (${kshot.k_shot}-shot) vs ${zero.arm} (0 labels), same ${state.result.n_query} queries`));
      box.appendChild(row);
    }
  }

  function renderPlane(state) {
    const empty = $("#plane-empty");
    const meta = $("#plane-meta");
    const proj = state.result && state.result.projection;

    if (!proj) {
      empty.hidden = false;
      meta.textContent = "";
      $("#legend").innerHTML = "";
      return;
    }
    empty.hidden = true;

    const s = state.session;
    const primary = Store.kShotVerdict() || Store.primaryVerdict();
    const points = [];
    s.support.forEach((c) => points.push({ id: c.image_id, label: c.label }));
    s.query.forEach((c, i) => {
      const pred = primary && primary.predictions.length > i ? primary.predictions[i] : undefined;
      points.push({
        id: c.image_id, label: c.label, pred,
        correct: c.label >= 0 && pred !== undefined ? c.label === pred : undefined,
      });
    });

    plane.setData(proj, { points });

    const fid = proj.margin_fidelity, sign = proj.sign_agreement;
    meta.innerHTML = "";
    const add = (label, value, cls) => {
      const s2 = el("span");
      s2.appendChild(el("b", null, label + " "));
      s2.appendChild(el("span", cls || "", value));
      meta.appendChild(s2);
    };
    add("arm", proj.arm);
    add("boundary agreement", `${(sign * 100).toFixed(0)}%`, sign >= 0.999 ? "good" : "soft");
    add("margin fidelity", fid.toFixed(3), fid >= 0.999 ? "good" : "soft");
    add("plane captures", `${(proj.explained_variance * 100).toFixed(0)}% of spread`);

    const legend = $("#legend");
    legend.innerHTML = "";
    const item = (color, shape, text) => {
      const wrap = el("span");
      const sw = el("span", `swatch ${shape}`);
      sw.style.background = color;
      wrap.appendChild(sw);
      wrap.appendChild(el("span", null, text));
      legend.appendChild(wrap);
    };
    const cs = getComputedStyle(document.documentElement);
    item(cs.getPropertyValue("--fresh"), "sq", "fresh prototype");
    item(cs.getPropertyValue("--rotten"), "sq", "rotten prototype");
    item(cs.getPropertyValue("--muted"), "", "filled = support · hollow = query · ✕ = misgraded");
  }

  function renderFooter(state) {
    const proj = state.result && state.result.projection;
    $("#footer-meta").textContent = proj
      ? `${proj.n_support} support · ${proj.n_query} query · ${state.result.n_labelled} labelled`
      : "";
  }

  /* --------------------------------------------------------------- go -- */
  document.addEventListener("DOMContentLoaded", boot);
})();
