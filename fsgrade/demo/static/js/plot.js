/* Canvas renderer for the decision plane.

   Points animate from their previous position to the new one, so adding a
   support image *moves* the prototypes rather than teleporting them — which is
   the whole point of the demo: you can watch the class centres form.

   The boundary is the vertical line u = 0. It is not an approximation: for
   every logit form the demo supports, the model's decision is exactly the sign
   of u. */
(function (global) {
  "use strict";

  const EASE = (t) => 1 - Math.pow(1 - t, 3);
  const DURATION = 520;

  class DecisionPlane {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.prev = new Map();     // id -> {x, y}
      this.current = [];
      this.anim = null;
      this.t0 = 0;
      this.colors = {};
      this._resize();
      global.addEventListener("resize", () => { this._resize(); this.redraw(1); });
    }

    _resize() {
      const dpr = global.devicePixelRatio || 1;
      const rect = this.canvas.getBoundingClientRect();
      const w = Math.max(rect.width || this.canvas.width, 320);
      const h = Math.round(w * 0.62);
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
      this.canvas.style.height = h + "px";
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.w = w;
      this.h = h;
    }

    _readColors() {
      const cs = getComputedStyle(document.documentElement);
      const get = (n, f) => (cs.getPropertyValue(n) || f).trim();
      this.colors = {
        fresh: get("--fresh", "#2c7a4b"),
        rotten: get("--rotten", "#a4501e"),
        rule: get("--rule", "#dbe2da"),
        rule2: get("--rule-2", "#c3cec4"),
        muted: get("--muted", "#6d7d73"),
        ink: get("--ink", "#16211c"),
        accent: get("--accent", "#1f6f62"),
        surface: get("--surface-2", "#eef2ed"),
      };
    }

    /* --------------------------------------------------------------- */
    setData(projection, meta) {
      if (!projection) { this.current = []; this.redraw(1); return; }
      this._readColors();

      const { coords, prototypes, bounds } = projection;
      const nSupport = projection.n_support || 0;
      const pad = 34;
      const uMin = bounds.u_min, uMax = bounds.u_max;
      const vMin = bounds.v_min, vMax = bounds.v_max;
      const sx = (u) => pad + ((u - uMin) / (uMax - uMin || 1)) * (this.w - 2 * pad);
      const sy = (v) => this.h - pad - ((v - vMin) / (vMax - vMin || 1)) * (this.h - 2 * pad);

      const points = coords.map((c, i) => {
        const isSupport = i < nSupport;
        const info = (meta && meta.points && meta.points[i]) || {};
        return {
          id: info.id || `${isSupport ? "s" : "q"}${i}`,
          x: sx(c[0]), y: sy(c[1]),
          role: isSupport ? "support" : "query",
          label: info.label,
          pred: info.pred,
          correct: info.correct,
        };
      });

      prototypes.forEach((p, c) => {
        points.push({ id: `proto${c}`, x: sx(p[0]), y: sy(p[1]), role: "prototype", label: c });
      });

      this.boundaryX = sx(0);
      this.axisY = sy(0);
      this.meta = meta || {};
      this.current = points;
      this._animate();
    }

    _animate() {
      if (this.anim) cancelAnimationFrame(this.anim);
      const reduce = global.matchMedia("(prefers-reduced-motion: reduce)").matches;
      if (reduce) { this.redraw(1); this._commit(); return; }
      this.t0 = performance.now();
      const step = (now) => {
        const t = Math.min(1, (now - this.t0) / DURATION);
        this.redraw(EASE(t));
        if (t < 1) this.anim = requestAnimationFrame(step);
        else { this.anim = null; this._commit(); }
      };
      this.anim = requestAnimationFrame(step);
    }

    _commit() {
      this.prev = new Map(this.current.map((p) => [p.id, { x: p.x, y: p.y }]));
    }

    /* --------------------------------------------------------------- */
    redraw(t = 1) {
      const ctx = this.ctx;
      if (!this.colors.rule) this._readColors();
      ctx.clearRect(0, 0, this.w, this.h);

      if (!this.current.length) return;

      this._drawRegions(ctx);
      this._drawAxes(ctx);

      // Draw queries first, then support, then prototypes on top.
      const order = { query: 0, support: 1, prototype: 2 };
      const sorted = [...this.current].sort((a, b) => order[a.role] - order[b.role]);
      for (const p of sorted) {
        const from = this.prev.get(p.id) || { x: this.boundaryX, y: this.axisY };
        const x = from.x + (p.x - from.x) * t;
        const y = from.y + (p.y - from.y) * t;
        this._drawPoint(ctx, p, x, y);
      }
      this._drawLabels(ctx);
    }

    _drawRegions(ctx) {
      const g = ctx.createLinearGradient(0, 0, this.w, 0);
      g.addColorStop(0, this._alpha(this.colors.fresh, 0.09));
      g.addColorStop(Math.max(0, Math.min(1, this.boundaryX / this.w)), this._alpha(this.colors.fresh, 0.02));
      g.addColorStop(Math.max(0, Math.min(1, this.boundaryX / this.w)), this._alpha(this.colors.rotten, 0.02));
      g.addColorStop(1, this._alpha(this.colors.rotten, 0.09));
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, this.w, this.h);
    }

    _drawAxes(ctx) {
      // Horizontal axis: the line through both prototypes.
      ctx.save();
      ctx.strokeStyle = this.colors.rule2;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 4]);
      ctx.beginPath();
      ctx.moveTo(0, this.axisY); ctx.lineTo(this.w, this.axisY);
      ctx.stroke();
      ctx.restore();

      // The decision boundary, u = 0. Solid and deliberate.
      ctx.save();
      ctx.strokeStyle = this.colors.ink;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(this.boundaryX, 12); ctx.lineTo(this.boundaryX, this.h - 12);
      ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.font = "600 10px ui-monospace, Menlo, Consolas, monospace";
      ctx.fillStyle = this.colors.muted;
      ctx.textAlign = "center";
      ctx.fillText("u = 0", this.boundaryX, this.h - 2);
      ctx.restore();
    }

    _drawPoint(ctx, p, x, y) {
      const fresh = this.colors.fresh, rotten = this.colors.rotten;

      if (p.role === "prototype") {
        const col = p.label === 1 ? rotten : fresh;
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(Math.PI / 4);
        ctx.fillStyle = col;
        ctx.strokeStyle = this.colors.surface;
        ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.rect(-8, -8, 16, 16); ctx.fill(); ctx.stroke();
        ctx.restore();

        ctx.save();
        ctx.shadowColor = this._alpha(col, 0.55);
        ctx.shadowBlur = 14;
        ctx.strokeStyle = this._alpha(col, 0.0);
        ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.stroke();
        ctx.restore();
        return;
      }

      const truth = p.label === 1 ? rotten : (p.label === 0 ? fresh : this.colors.muted);

      if (p.role === "support") {
        ctx.fillStyle = truth;
        ctx.strokeStyle = this.colors.surface;
        ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        return;
      }

      // Query: hollow ring coloured by prediction; a cross marks a miss.
      const predCol = p.pred === 1 ? rotten : fresh;
      ctx.strokeStyle = predCol;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, 4.6, 0, Math.PI * 2); ctx.stroke();

      if (p.correct === false) {
        ctx.save();
        ctx.strokeStyle = this.colors.ink;
        ctx.globalAlpha = 0.75;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.moveTo(x - 7, y - 7); ctx.lineTo(x + 7, y + 7);
        ctx.moveTo(x + 7, y - 7); ctx.lineTo(x - 7, y + 7);
        ctx.stroke();
        ctx.restore();
      }
    }

    _drawLabels(ctx) {
      ctx.save();
      ctx.font = "600 11px ui-sans-serif, system-ui, sans-serif";
      ctx.textBaseline = "top";
      ctx.globalAlpha = 0.8;
      ctx.fillStyle = this.colors.fresh;
      ctx.textAlign = "left";
      ctx.fillText("← fresh", 10, 9);
      ctx.fillStyle = this.colors.rotten;
      ctx.textAlign = "right";
      ctx.fillText("rotten →", this.w - 10, 9);
      ctx.restore();
    }

    _alpha(color, a) {
      const c = color.trim();
      if (c.startsWith("#")) {
        const hex = c.length === 4
          ? c.slice(1).split("").map((ch) => ch + ch).join("")
          : c.slice(1);
        const n = parseInt(hex, 16);
        return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
      }
      return c;
    }

    reset() {
      this.prev.clear();
      this.current = [];
      this.ctx.clearRect(0, 0, this.w, this.h);
    }
  }

  global.DecisionPlane = DecisionPlane;
})(window);
