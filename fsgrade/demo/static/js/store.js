/* Central state with a revision guard.
   The backend stamps every mutating response with session.revision; anything
   older than what we already hold is a stale reply and gets dropped. */
(function (global) {
  "use strict";

  const state = {
    ready: false,
    health: null,
    session: null,
    speciesList: [],
    methods: [],
    result: null,        // last prediction payload
    busy: false,
    banner: null,        // {text, kind}
    kappa: 5,
    kappaCalibrated: null,
    kappaSource: "",
  };

  const listeners = new Set();

  function notify() {
    for (const fn of listeners) {
      try { fn(state); } catch (err) { console.error("listener failed", err); }
    }
  }

  const Store = {
    get state() { return state; },

    subscribe(fn) {
      listeners.add(fn);
      fn(state);
      return () => listeners.delete(fn);
    },

    set(patch) {
      Object.assign(state, patch);
      notify();
    },

    /* Accept a session only if it is at least as new as the one we hold. */
    setSession(session) {
      if (!session) return false;
      const current = state.session;
      if (current && current.session_id === session.session_id
          && session.revision < current.revision) {
        return false;                       // stale reply, discard
      }
      state.session = session;
      notify();
      return true;
    },

    setResult(payload) {
      if (!payload) return;
      if (payload.session) this.setSession(payload.session);
      state.result = payload;
      notify();
    },

    banner(text, kind = "info") {
      state.banner = text ? { text, kind } : null;
      notify();
    },

    clearBanner() { this.banner(null); },

    busy(flag) {
      state.busy = !!flag;
      notify();
    },

    /* Whichever arm carries the geometry we are drawing. */
    primaryVerdict() {
      const r = state.result;
      if (!r || !r.verdicts.length) return null;
      const armed = r.projection && r.projection.arm;
      return r.verdicts.find((v) => v.arm === armed) || r.verdicts[0];
    },

    zeroShotVerdict() {
      const r = state.result;
      if (!r) return null;
      return r.verdicts.find((v) => !v.uses_support) || null;
    },

    kShotVerdict() {
      const r = state.result;
      if (!r) return null;
      return r.verdicts.find((v) => v.uses_support) || null;
    },
  };

  global.Store = Store;
})(window);
