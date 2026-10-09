"use strict";

(() => {
  // Remove the capability from the address bar before loading any remote SDK.
  const tokenKey = "apm.apple_music.api_token";
  let apiToken = null;
  let linkCleared = true;
  try {
    const fragment = new URLSearchParams(window.location.hash.slice(1));
    if (fragment.has("token")) {
      apiToken = fragment.get("token");
      try { sessionStorage.setItem(tokenKey, apiToken); } catch (_) { /* This visit still works. */ }
    } else {
      try { apiToken = sessionStorage.getItem(tokenKey); } catch (_) { /* A fresh link is required. */ }
    }
  } finally {
    try {
      if (window.location.hash) history.replaceState(null, "", window.location.pathname + window.location.search);
    } catch (_) { linkCleared = false; }
  }

  const element = (id) => document.getElementById(id);
  const ui = {
    connection: element("connection"), connectionTitle: element("connection-title"),
    connectionDetail: element("connection-detail"), connect: element("connect"),
    disconnect: element("disconnect"), reload: element("reload-setup"),
    title: element("track-title"), artist: element("track-artist"), artwork: element("artwork"),
    placeholder: element("artwork-placeholder"), playbackLabel: element("playback-label"),
    playbackState: element("playback-state"), start: element("start-playback"),
    pause: element("pause-playback"), playerNote: element("player-note"), appleLink: element("apple-link"),
    form: element("song-form"), song: element("song-title"), songArtist: element("song-artist"),
    submit: element("request-song"), result: element("result-message"), candidates: element("candidates"),
    diagnostics: element("auth-diagnostics"), diagnosticOutput: element("auth-diagnostics-output"),
    playbackDiagnostics: element("playback-diagnostics"), playbackDiagnosticOutput: element("playback-diagnostics-output"),
  };
  const state = {
    music: null, ready: false, connecting: false, disconnecting: false,
    sessionId: null, generation: 0, pollAbort: null, requestedTrack: null,
    queueReady: false, manualStarting: false, requestBusy: false, requestNumber: 0,
    playbackErrors: 0, artworkURL: null, processed: new Set(),
    commandAbort: null, manualAbort: null, playChain: Promise.resolve(), playEpoch: 0,
    playInFlight: false, pauseHold: false, holdPauseBusy: false,
    pendingStarts: new Set(), queueOperation: Promise.resolve(),
    pendingPauses: new Set(), pauseDispatch: null,
    lastSDKPlay: -Infinity, lastSDKPause: -Infinity,
  };
  let sdkPromise = null;
  let initializing = false;

  class APIError extends Error {
    constructor(message, status = 0) { super(message); this.status = status; }
  }
  class UIError extends Error {}
  class PlaybackFailure extends Error {
    constructor(reason) { super("Playback could not be confirmed"); this.reason = reason; }
  }

  const playbackSDKReasons = new Set([
    "ACCESS_DENIED", "AUTHORIZATION_ERROR", "CONTENT_EQUIVALENT", "CONTENT_RESTRICTED",
    "CONTENT_UNAVAILABLE", "CONTENT_UNSUPPORTED", "DEVICE_LIMIT", "GEO_BLOCK",
    "MEDIA_CERTIFICATE", "MEDIA_DESCRIPTOR", "MEDIA_LICENSE", "MEDIA_KEY", "MEDIA_PLAYBACK",
    "MEDIA_SESSION", "NETWORK_ERROR", "NOT_FOUND", "OUTPUT_RESTRICTED", "SERVER_ERROR",
    "SERVICE_UNAVAILABLE", "STREAM_UPSELL", "SUBSCRIPTION_ERROR", "TOKEN_EXPIRED",
    "UNAUTHORIZED_ERROR", "UNSUPPORTED_ERROR", "USER_INTERACTION_REQUIRED", "WIDEVINE_CDM_EXPIRED",
  ]);
  const playbackStates = ["none", "loading", "playing", "paused", "stopped", "ended", "seeking", "waiting", "stalled", "completed"];
  const playbackMessages = {
    expired: "The playback request expired before it could start.",
    cancelled: "The playback request was cancelled.",
    authorization_lost: "Apple Music authorization could not be confirmed. Connect again.",
    queue_rejected: "Apple Music could not load the requested song.",
    queue_unavailable: "Apple Music did not provide a playable queue.",
    queue_mismatch: "The requested recording was not in the expected queue.",
    autoplay_blocked: "The browser requires a click. Use Start playback to begin the loaded song.",
    play_rejected: "Apple Music rejected the playback attempt.",
    media_error: "Apple Music reported a media playback error.",
    timeout: "Apple Music did not confirm playback before the request timed out.",
    unknown: "Playback could not be confirmed. Check the player before trying again.",
  };

  function playbackTrace(started) {
    return { started, phaseStarted: started, phase: "preflight", reason: "unknown", trackId: null };
  }
  function playbackPhase(trace, phase) { trace.phase = phase; trace.phaseStarted = performance.now(); }
  function playbackFailure(trace, error, fallback = "unknown") {
    if (trace.failed) return;
    trace.failed = true;
    trace.reason = error instanceof PlaybackFailure ? error.reason : fallback;
    // Only fixed SDK codes are read. Raw messages, response bodies and media
    // metadata never enter either the local display or completion diagnostics.
    try {
      const reason = [error?.reason, error?.errorCode, error?.code].find((value) =>
        typeof value === "string" && playbackSDKReasons.has(value));
      if (reason) trace.sdk_reason = reason;
      if (reason === "USER_INTERACTION_REQUIRED" || error?.name === "NotAllowedError") trace.reason = "autoplay_blocked";
      else if (["AUTHORIZATION_ERROR", "TOKEN_EXPIRED", "UNAUTHORIZED_ERROR"].includes(reason)) trace.reason = "authorization_lost";
    } catch (_) { /* Unknown/throwing SDK properties keep the fixed fallback. */ }
  }
  function playbackSnapshot(trace) {
    const now = performance.now();
    const milliseconds = (value) => Math.min(60000, Math.max(0, Math.round(value)));
    const snapshot = { phase: trace.phase, reason: trace.reason,
      elapsed_ms: milliseconds(now - trace.started), phase_ms: milliseconds(now - trace.phaseStarted),
      state: "unknown", track_matches: null };
    if (trace.sdk_reason) snapshot.sdk_reason = trace.sdk_reason;
    if (trace.queue_check) snapshot.queue_check = trace.queue_check;
    try {
      const states = window.MusicKit?.PlaybackStates;
      snapshot.state = playbackStates.find((name) => typeof states?.[name] === "number" && state.music?.playbackState === states[name]) || "unknown";
      const observed = observedTrackID();
      if (trace.trackId && observed) snapshot.track_matches = trace.trackId === observed;
    } catch (_) { /* Keep unknown state without inspecting SDK errors. */ }
    // Retain a local snapshot even when cancellation prevents acknowledgment.
    try {
      ui.playbackDiagnostics.hidden = false;
      ui.playbackDiagnosticOutput.textContent = JSON.stringify(snapshot, null, 2);
    } catch (_) { /* A diagnostic display must not prevent acknowledgment. */ }
    return snapshot;
  }
  function playbackCurrentFailure(current, deadline, initial = false, requireAuthorization = true) {
    if (requireAuthorization && !authorized()) return new PlaybackFailure("authorization_lost");
    if (!current()) return new PlaybackFailure("cancelled");
    if (!Number.isFinite(deadline) || performance.now() >= deadline) return new PlaybackFailure(initial ? "expired" : "timeout");
    return null;
  }

  // Only known SDK reasons and selected HTTP statuses can reach the page.
  // MKError.reason/errorCode and responseError.data.status are defined by the
  // official v3 SDK; descriptions, messages, URLs, and response bodies are not.
  const sdkReasons = new Set([
    "ACCESS_DENIED", "AGE_GATE", "AUTHORIZATION_ERROR", "CONFIGURATION_ERROR",
    "CONTENT_RESTRICTED", "DEVICE_LIMIT", "GEO_BLOCK", "INVALID_ARGUMENTS",
    "NETWORK_ERROR", "NOT_FOUND", "QUOTA_EXCEEDED", "SERVER_ERROR",
    "SERVICE_UNAVAILABLE", "STREAM_UPSELL", "SUBSCRIPTION_ERROR", "TOKEN_EXPIRED",
    "UNAUTHORIZED_ERROR", "UNKNOWN_ERROR", "UNSUPPORTED_ERROR", "USER_INTERACTION_REQUIRED",
  ]);
  const diagnosticStatuses = new Set([400, 401, 403, 404, 408, 409, 429, 500, 502, 503, 504]);

  // Opt-in, page-local troubleshooting. No token, account data, callback params,
  // raw errors, or network bodies are read into this trace or sent to a server.
  const diagnosticsEnabled = new URLSearchParams(window.location.search).get("diagnostics") === "1";
  const authorizationStatuses = new Set([-1, 0, 1, 2, 3]);
  const appleMethods = new Map([
    ["authorize", "apple_authorize"], ["decline", "apple_decline"],
    ["unavailable", "apple_unavailable"], ["switchUserId", "apple_switch_user"],
    ["close", "apple_close"], ["thirdPartyInfo", "apple_third_party_info"],
  ]);
  let authTrace = null;

  function diagnosticEvent(label, error) {
    // SDK subscribers run synchronously: diagnostics must never throw into SDK.
    try {
      if (!diagnosticsEnabled || !authTrace?.active) return;
      const elapsed = Math.min(3600000, Math.max(0, Math.round(performance.now() - authTrace.started)));
      const line = `${elapsed}ms ${label}${error ? diagnosticSuffix(error) : ""}`;
      // Keep the beginning and latest event even if a popup floods the observer.
      if (authTrace.lines.length < 40) authTrace.lines.push(line);
      else authTrace.lines[39] = line;
      authTrace.seen.add(label);
      ui.diagnosticOutput.textContent = authTrace.lines.join("\n");
    } catch (_) { /* Never interrupt authorization for a diagnostic failure. */ }
  }

  function diagnosticStatus(value, prefix = "status") {
    if (Number.isInteger(value) && authorizationStatuses.has(value)) diagnosticEvent(`${prefix}_${value}`);
  }

  function beginAuthDiagnostics() {
    try {
      if (!diagnosticsEnabled) return;
      authTrace = { active: true, started: performance.now(), lines: [], seen: new Set() };
      diagnosticEvent("authorization_started");
      diagnosticStatus(state.music?.authorizationStatus, "initial_status");
    } catch (_) { /* Optional diagnostics cannot prevent sign-in. */ }
  }

  function finishAuthDiagnostics() {
    try {
      if (!authTrace?.active) return;
      diagnosticStatus(state.music?.authorizationStatus, "final_status");
      authTrace.active = false;
      const seen = authTrace.seen;
      let summary = "The trace records observed events, not account eligibility or confirmed playback.";
      if (seen.has("authorization_rejected")) {
        if (seen.has("apple_unavailable") || seen.has("status_-1")) {
          summary = "Apple reported authorization unavailable; the underlying Apple response is still needed to explain why.";
        } else if (seen.has("apple_decline") || seen.has("status_1")) {
          summary = "Apple reported authorization declined. This does not establish which action or account condition caused it.";
        } else if (seen.has("apple_switch_user")) {
          summary = "Apple requested an account switch instead of completing authorization.";
        } else if (seen.has("status_2") || seen.has("status_3")) {
          summary = "MusicKit reached an authorized status before rejecting completion; this does not prove token validity.";
        } else if (seen.has("apple_authorize")) {
          summary = "Apple's authorization callback reached this page, but MusicKit failed to finish authorization.";
        } else {
          summary = "No recognized Apple authorization callback was observed before MusicKit rejected authorization. The cause is not yet known.";
        }
      }
      ui.diagnosticOutput.textContent = `${summary}\n\n${authTrace.lines.join("\n")}\n\nStatus 0 can be cleanup after failure; it does not identify the cause.`;
    } catch (_) { /* Preserve sign-in behavior if a diagnostic cannot render. */ }
    finally { if (authTrace) authTrace.active = false; }
  }

  ui.diagnostics.hidden = !diagnosticsEnabled;
  if (diagnosticsEnabled) {
    ui.diagnosticOutput.textContent = "Ready to trace the next Connect attempt. Only event names, status codes, and elapsed times appear here.";
    // This private SDK protocol is observed passively, never used to authorize
    // or control playback. Source-verified against Apple's official v3 SDK.
    window.addEventListener("message", (event) => {
      try {
        if (!authTrace?.active || event.origin !== "https://authorize.music.apple.com") return;
        const data = event.data;
        if (!data || typeof data !== "object" || Array.isArray(data) || data.jsonrpc !== "2.0") return;
        const label = appleMethods.get(data.method);
        if (label) diagnosticEvent(label);
      } catch (_) { /* Malformed messages cannot affect MusicKit. */ }
    });
  }

  function safeErrorFacts(error) {
    try {
      const reasons = [error?.reason, error?.errorCode, error?.code, error?.name];
      const statuses = [error, error?.status, error?.statusCode, error?.code, error?.response?.status, error?.data?.status];
      return {
        reason: reasons.find((value) => typeof value === "string" && sdkReasons.has(value)),
        status: statuses.find((value) => Number.isInteger(value) && diagnosticStatuses.has(value)),
      };
    } catch (_) { return {}; }
  }

  function diagnosticSuffix(error, source = "Apple Music") {
    const { reason, status } = safeErrorFacts(error);
    const parts = [reason, status ? `HTTP ${status}` : null].filter(Boolean);
    return parts.length ? ` (${source}: ${parts.join("; ")})` : "";
  }

  function connectFailure(error, phase) {
    const { reason, status } = safeErrorFacts(error);
    if (phase === "local") {
      const message = status === 401
        ? "Apple Music sign-in succeeded, but the local APM player link was rejected. Open a fresh player link from APM."
        : "Apple Music sign-in succeeded, but APM could not register this browser. Check that the local APM server is running, then connect again.";
      connection("error", "APM player connection failed", message + diagnosticSuffix(error, "Local API"));
      return;
    }
    if (phase === "region") {
      connection("error", "Apple Music region unavailable",
        "Apple Music sign-in succeeded, but the account region could not be read. Disconnect and reconnect Apple Music." + diagnosticSuffix(error));
      return;
    }
    if (phase === "playback_setup") {
      state.ready = false;
      connection("error", "Player unavailable", (error instanceof UIError ? error.message : "Full-song playback could not be enabled. Reload player to try again.") + diagnosticSuffix(error));
      return;
    }
    let message = "Apple Music sign-in could not finish. Allow the Apple Music sign-in window and try again. If it continues, reload the player to load the latest APM music code and configuration.";
    if (status === 401 || ["UNAUTHORIZED_ERROR", "TOKEN_EXPIRED"].includes(reason)) {
      message = "Apple Music rejected authorization. Reload player, then connect again. If it continues, check APM’s Apple Music credentials.";
    } else if (status === 403 || reason === "ACCESS_DENIED") {
      message = "Apple Music denied access. Check the account authorization and APM’s Apple Music setup, then connect again.";
    } else if (["SUBSCRIPTION_ERROR", "STREAM_UPSELL"].includes(reason)) {
      message = "Apple Music reported a subscription requirement. Check that the signed-in account has an active Apple Music subscription.";
    } else if (reason === "USER_INTERACTION_REQUIRED") {
      message = "The browser requires a sign-in gesture. Click Connect Apple Music and allow its sign-in window.";
    } else if (["NETWORK_ERROR", "SERVER_ERROR", "SERVICE_UNAVAILABLE", "QUOTA_EXCEEDED"].includes(reason) || [429, 500, 502, 503, 504].includes(status)) {
      message = "Apple Music could not complete the request. Check your internet connection, wait a moment, then connect again.";
    }
    connection("error", "Apple Music sign-in failed", message + diagnosticSuffix(error));
  }

  async function api(path, { method = "GET", body, signal, timeout = 25000 } = {}) {
    if (!apiToken) throw new APIError("Open the player link printed by APM to connect.", 401);
    const controller = new AbortController();
    const abort = () => controller.abort();
    if (signal?.aborted) controller.abort();
    signal?.addEventListener("abort", abort, { once: true });
    const timer = window.setTimeout(abort, timeout);
    try {
      const headers = { Authorization: `Bearer ${apiToken}` };
      if (body !== undefined) headers["Content-Type"] = "application/json";
      const response = await fetch(path, {
        method, headers, body: body === undefined ? undefined : JSON.stringify(body),
        signal: controller.signal, cache: "no-store", credentials: "same-origin",
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const message = response.status === 401
          ? "Open a fresh player link from APM to reconnect."
          : (typeof payload.detail === "string" ? payload.detail : "The local player request could not be completed.");
        throw new APIError(message, response.status);
      }
      return payload;
    } catch (error) {
      if (signal?.aborted) throw new DOMException("Request cancelled", "AbortError");
      if (error instanceof APIError) throw error;
      throw new APIError("The local APM server did not respond. Check that it is running.");
    } finally {
      window.clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
    }
  }

  function connection(tone, title, detail) {
    ui.connection.dataset.state = tone;
    ui.connectionTitle.textContent = title;
    ui.connectionDetail.textContent = detail;
  }

  function result(message, tone = "neutral") {
    ui.result.textContent = message;
    ui.result.dataset.tone = tone;
  }

  function authorized() { return Boolean(state.music?.isAuthorized); }
  function playing() {
    return Boolean(state.music && window.MusicKit?.PlaybackStates &&
      state.music.playbackState === window.MusicKit.PlaybackStates.playing);
  }
  function pauseActivity() {
    const states = window.MusicKit?.PlaybackStates;
    if (!states || !state.music) return null;
    const value = state.music.playbackState;
    if (["playing", "loading", "seeking", "waiting", "stalled"].some((name) =>
      typeof states[name] === "number" && value === states[name])) return true;
    if (["none", "paused", "stopped", "ended", "completed"].some((name) =>
      typeof states[name] === "number" && value === states[name])) return false;
    return null;
  }

  function playbackChanged() {
    // A pause intent also covers a startup Promise that completes after its
    // command was cancelled. Only a subsequent explicit play/resume releases it.
    if (state.pauseHold && playing() && !state.holdPauseBusy) {
      state.holdPauseBusy = true;
      try {
        const music = state.music;
        dispatchSDKPause(() => state.music === music && state.pauseHold && playing(), performance.now() + 1200, music)
          .catch(() => {}).finally(() => { state.holdPauseBusy = false; });
      } catch (_) { state.holdPauseBusy = false; }
    }
    renderPlayer();
  }
  function observedTrackID() {
    const id = state.music?.nowPlayingItem?.id;
    return id !== undefined && /^\d+$/.test(String(id)) ? String(id) : null;
  }
  function queueCheck(queue, trackId) {
    if (!queue) return "absent";
    if (queue !== state.music?.queue) return "different_queue";
    if (!Array.isArray(queue.items) || queue.items.length !== 1) return "wrong_length";
    return String(queue.items[0]?.id) === trackId ? "matched" : "wrong_track";
  }
  function queueMatches(queue, trackId) { return queueCheck(queue, trackId) === "matched"; }

  function controls() {
    const connected = Boolean(state.sessionId) && authorized();
    ui.connect.disabled = !state.ready || state.connecting || state.disconnecting;
    ui.connect.hidden = connected;
    ui.disconnect.hidden = !connected && !authorized();
    ui.disconnect.disabled = state.connecting || state.disconnecting;
    ui.reload.disabled = initializing || state.connecting || state.disconnecting || Boolean(state.sessionId) || playing();
    ui.song.disabled = !connected || state.requestBusy;
    ui.songArtist.disabled = !connected || state.requestBusy;
    ui.submit.disabled = !connected || state.requestBusy;
    ui.start.disabled = !authorized() || !state.requestedTrack || !state.queueReady ||
      !queueMatches(state.music?.queue, state.requestedTrack) || state.manualStarting || playing() || state.disconnecting;
    ui.pause.disabled = !authorized() || !playing() || state.disconnecting;
    ui.candidates.querySelectorAll("button").forEach((button) => { button.disabled = !connected || state.requestBusy; });
  }

  function artworkURL(value) {
    if (typeof value !== "string") return null;
    try {
      const url = new URL(value.replaceAll("{w}", "480").replaceAll("{h}", "480").replaceAll("{f}", "jpg"));
      if (url.protocol !== "https:" || !(url.hostname.endsWith(".mzstatic.com") || url.hostname.endsWith(".apple.com"))) return null;
      return url.href;
    } catch (_) { return null; }
  }

  function appleURL(value) {
    try {
      const url = new URL(value);
      return url.protocol === "https:" && url.hostname === "music.apple.com" ? url.href : "https://music.apple.com/";
    } catch (_) { return "https://music.apple.com/"; }
  }

  function renderPlayer() {
    const item = state.music?.nowPlayingItem;
    const attributes = item?.attributes || {};
    ui.title.textContent = typeof attributes.name === "string" ? attributes.name : "Choose something to play";
    ui.artist.textContent = typeof attributes.artistName === "string" ? attributes.artistName : "Your next song will appear here.";
    const artwork = artworkURL(attributes.artwork?.url);
    if (artwork !== state.artworkURL) {
      state.artworkURL = artwork;
      ui.artwork.hidden = true;
      ui.placeholder.hidden = false;
      if (artwork) {
        ui.artwork.alt = `${ui.title.textContent} artwork`;
        ui.artwork.src = artwork;
      } else {
        ui.artwork.removeAttribute("src");
        ui.artwork.alt = "";
      }
    }
    ui.appleLink.href = appleURL(attributes.url);
    const isPlaying = playing();
    ui.playbackLabel.dataset.playing = String(isPlaying);
    const playback = state.music?.playbackState;
    const states = window.MusicKit?.PlaybackStates;
    ui.playbackState.textContent = isPlaying ? "Playing" :
      (states && playback === states.paused ? "Paused" : (state.requestedTrack && state.queueReady ? "Ready to play" : "Not playing"));
    if (state.requestedTrack && observedTrackID() === state.requestedTrack) state.queueReady = true;
    if (isPlaying) ui.playerNote.textContent = "Playing through this browser on your Mac.";
    controls();
  }

  ui.artwork.addEventListener("load", () => { ui.artwork.hidden = false; ui.placeholder.hidden = true; });
  ui.artwork.addEventListener("error", () => { ui.artwork.hidden = true; ui.placeholder.hidden = false; });

  function loadSDK() {
    if (sdkPromise) return sdkPromise;
    sdkPromise = new Promise((resolve, reject) => {
      if (window.MusicKit) { resolve(window.MusicKit); return; }
      let settled = false;
      const finish = () => {
        if (settled || !window.MusicKit) return;
        settled = true;
        window.clearTimeout(timer);
        document.removeEventListener("musickitloaded", finish);
        resolve(window.MusicKit);
      };
      const fail = () => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        document.removeEventListener("musickitloaded", finish);
        sdkPromise = null;
        reject(new UIError("Apple Music could not load. Check your connection, then reload the player."));
      };
      const timer = window.setTimeout(fail, 20000);
      document.addEventListener("musickitloaded", finish);
      const script = document.createElement("script");
      script.src = "https://js-cdn.music.apple.com/musickit/v3/musickit.js";
      script.async = true;
      script.addEventListener("load", finish, { once: true });
      script.addEventListener("error", fail, { once: true });
      document.head.appendChild(script);
    });
    return sdkPromise;
  }

  function requireFullPlayback(music, MusicKit) {
    // Source-verified in Apple's official v3 SDK, rather than a documented
    // subscription-status API. Fail closed if this compatibility guard changes.
    const mode = MusicKit.PlaybackMode?.FULL_PLAYBACK_ONLY;
    if (mode === undefined || !("playbackMode" in music)) {
      throw new UIError("This MusicKit version cannot enforce full-song playback. Update APM before connecting.");
    }
    music.playbackMode = mode;
    if (music.playbackMode !== mode) {
      throw new UIError("Full-song playback could not be enabled. Update APM before connecting.");
    }
  }

  async function initialize() {
    if (initializing || state.connecting || state.disconnecting || state.sessionId || playing()) return;
    initializing = true;
    state.ready = false;
    ui.reload.disabled = true;
    ui.reload.hidden = true;
    connection("loading", "Checking setup…", "Preparing the local player.");
    controls();
    try {
      if (!linkCleared) throw new UIError("The player link could not be cleared from the address bar. Reopen it from APM.");
      if (!apiToken) throw new UIError("Open the player link printed by APM to connect.");
      const config = await api("/v1/config", { timeout: 10000 });
      if (config.configured === false || typeof config.developer_token !== "string" || !config.developer_token) {
        connection("error", "Apple Music needs setup", typeof config.message === "string" ? config.message : "Complete Apple Music setup in APM, then reload the player here.");
        ui.reload.hidden = false;
        result("Finish Apple Music setup in APM to connect this player.");
        return;
      }
      const MusicKit = await loadSDK();
      // developerToken is read-only. Public MusicKit.configure creates a fresh
      // instance with this token and cleans up the previous instance in v3.
      // Refresh is explicit and blocked while connected or playing.
      const previousMusic = state.music;
      state.music = null;
      state.requestedTrack = null;
      state.queueReady = false;
      if (previousMusic) {
        try { await beforeDeadline(dispatchSDKPause(() => state.music === null, performance.now() + 2000, previousMusic), performance.now() + 2000); } catch (_) { /* Configure performs SDK cleanup. */ }
      }
      state.music = await MusicKit.configure({ developerToken: config.developer_token,
        app: { name: typeof config.app_name === "string" ? config.app_name : "APM Assistant", build: "0.1.0" } });
      requireFullPlayback(state.music, MusicKit);
      state.music.addEventListener("playbackStateDidChange", playbackChanged);
      state.music.addEventListener("nowPlayingItemDidChange", renderPlayer);
      state.music.addEventListener("queueItemsDidChange", renderPlayer);
      state.music.addEventListener("mediaPlaybackError", () => {
        state.playbackErrors += 1;
        ui.playerNote.textContent = "Playback could not be confirmed. Check your Apple Music access, then try Start playback.";
        renderPlayer();
      });
      state.music.addEventListener("authorizationStatusDidChange", (event) => {
        try { diagnosticStatus(event?.authorizationStatus); } catch (_) { /* Never throw into SDK. */ }
        if (!authorized() && state.sessionId && !state.disconnecting) {
          const oldSession = state.sessionId;
          loseConnection("Apple Music authorization ended. Connect again to continue.");
          api("/v1/player/disconnect", { method: "POST", body: { session_id: oldSession }, timeout: 3000 }).catch(() => {});
        }
        controls();
      });
      state.ready = true;
      connection("ready", "Ready to connect", "Sign in to Apple Music to use your subscription on this Mac.");
      result("Connect Apple Music, then request a song.");
    } catch (error) {
      state.ready = false;
      connection("error", "Player unavailable", error instanceof APIError || error instanceof UIError ? error.message :
        "Apple Music could not initialize. Check APM’s Apple Music setup, then reload the player." + diagnosticSuffix(error));
      ui.reload.hidden = false;
    } finally {
      initializing = false;
      ui.reload.disabled = false;
      controls();
    }
  }

  function stopPolling() {
    state.generation += 1;
    state.playEpoch += 1;
    state.commandAbort?.abort();
    state.manualAbort?.abort();
    state.pollAbort?.abort();
    state.pollAbort = null;
  }

  function loseConnection(message) {
    stopPolling();
    state.sessionId = null;
    state.requestNumber += 1;
    state.requestBusy = false;
    connection("error", "Player disconnected", message);
    controls();
  }

  async function connect() {
    if (!state.ready || state.connecting || state.disconnecting) return;
    state.connecting = true;
    let phase = "authorization";
    connection("loading", "Connecting Apple Music…", "Complete sign-in in the Apple Music window.");
    controls();
    beginAuthDiagnostics();
    try {
      // Keep authorize directly in the click handler's call stack.
      await state.music.authorize();
      diagnosticEvent("authorization_resolved");
      if (!authorized()) {
        connection("ready", "Ready to connect", "Sign-in was not completed. Choose Connect Apple Music to try again.");
        ui.reload.hidden = false;
        return;
      }
      phase = "playback_setup";
      requireFullPlayback(state.music, window.MusicKit);
      phase = "region";
      const storefront = String(state.music.storefrontId || "").toLowerCase();
      if (!/^[a-z]{2}$/.test(storefront)) throw new UIError("Apple Music did not provide a supported account region. Try connecting again.");
      phase = "local";
      diagnosticEvent("local_registration_started");
      const sessionId = crypto.randomUUID();
      await api("/v1/player/session", { method: "POST", body: { session_id: sessionId, storefront, protocol_version: 3 }, timeout: 10000 });
      diagnosticEvent("local_registration_succeeded");
      stopPolling();
      state.sessionId = sessionId;
      state.processed.clear();
      state.requestedTrack = null;
      state.queueReady = false;
      ui.reload.hidden = true;
      connection("connected", "Apple Music connected", `Your ${storefront.toUpperCase()} catalog is ready. Keep this tab open for APM requests.`);
      result("Request a song below, or ask APM by voice.");
      const generation = state.generation;
      pollCommands(sessionId, generation);
    } catch (error) {
      diagnosticEvent(phase === "authorization" ? "authorization_rejected" : `${phase}_failed`, error);
      connectFailure(error, phase);
      ui.reload.hidden = false;
    } finally {
      finishAuthDiagnostics();
      state.connecting = false;
      renderPlayer();
    }
  }

  const delay = (milliseconds) => new Promise((resolve) => window.setTimeout(resolve, milliseconds));

  async function beforeDeadline(promise, deadline, signal) {
    const remaining = deadline - performance.now();
    if (remaining <= 0) throw new PlaybackFailure("timeout");
    let timer;
    let abort;
    try {
      return await Promise.race([Promise.resolve(promise), new Promise((_, reject) => {
        timer = window.setTimeout(() => reject(new PlaybackFailure("timeout")), remaining);
        abort = () => reject(new PlaybackFailure("cancelled"));
        if (signal?.aborted) abort();
        else signal?.addEventListener("abort", abort, { once: true });
      })]);
    } finally { window.clearTimeout(timer); signal?.removeEventListener("abort", abort); }
  }

  // Apple's official v3 SDK wraps public play and pause independently in
  // AsyncDebounce(250, {isImmediate:true}). A second same-method call inside
  // that window silently does nothing and extends the window. Gate actual
  // invocations (5ms scheduling margin), not commands, and never retry play.
  // Source: https://js-cdn.music.apple.com/musickit/v3/musickit.js
  const sdkControlWindow = 255;
  function dispatchSDKPause(current, deadline, music = state.music) {
    const request = { current, deadline };
    const existing = state.pauseDispatch;
    if (existing && existing.music === music) {
      existing.requests.push(request);
      return existing.promise;
    }
    const batch = { music, requests: [request], promise: null };
    let resolve;
    let reject;
    batch.promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    state.pauseDispatch = batch;
    state.pendingPauses.add(batch);
    const finish = (error) => {
      if (state.pauseDispatch === batch) state.pauseDispatch = null;
      state.pendingPauses.delete(batch);
      if (error) reject(error); else resolve();
    };
    const run = () => {
      const now = performance.now();
      const currentRequests = batch.requests.filter((entry) => {
        try { return entry.current(); }
        catch (_) { return false; }
      });
      const active = currentRequests.filter((entry) => Number.isFinite(entry.deadline) && now < entry.deadline);
      if (!active.length) return finish(new PlaybackFailure(currentRequests.length ? "timeout" : "cancelled"));
      if (state.music === music && pauseActivity() === false && !state.pendingStarts.size) return finish();
      const remaining = state.lastSDKPause + sdkControlWindow - Date.now();
      if (remaining > 0) {
        // Recheck each request's deadline and intent when the cooldown ends.
        window.setTimeout(run, Math.max(1, Math.min(remaining, Math.max(...active.map((entry) => entry.deadline)) - now)));
        return;
      }
      try {
        state.lastSDKPause = Date.now();
        Promise.resolve(music.pause()).then(() => finish(), finish);
      } catch (error) { finish(error); }
    };
    run();
    return batch.promise;
  }

  function invokeSDKPlay(current, deadline, signal, beforeInvoke, afterInvoke, waiting) {
    const music = state.music;
    const invoke = () => {
      const error = playbackCurrentFailure(current, deadline);
      if (error || signal?.aborted || state.music !== music) throw error || new PlaybackFailure("cancelled");
      state.lastSDKPlay = Date.now();
      beforeInvoke();
      const operation = Promise.resolve(music.play());
      afterInvoke?.();
      return operation;
    };
    const pauses = [...state.pendingPauses].filter((batch) => batch.music === music).map((batch) => batch.promise);
    if (!pauses.length && Date.now() >= state.lastSDKPlay + sdkControlWindow) return invoke();
    waiting?.();
    return (async () => {
      while (true) {
        const error = playbackCurrentFailure(current, deadline);
        if (error || signal?.aborted || state.music !== music) throw error || new PlaybackFailure("cancelled");
        const pending = [...state.pendingPauses].filter((batch) => batch.music === music).map((batch) => batch.promise);
        if (pending.length) { await beforeDeadline(Promise.allSettled(pending), deadline, signal); continue; }
        const remaining = state.lastSDKPlay + sdkControlWindow - Date.now();
        if (remaining <= 0) break;
        await beforeDeadline(delay(remaining), deadline, signal);
      }
      return invoke();
    })();
  }

  function pauseAndConfirm(current, deadline, wasPlaying, trace) {
    const music = state.music;
    return new Promise((resolve, reject) => {
      let settled = false;
      let timer;
      const finish = (error) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        for (const [name, callback] of [["playbackStateDidChange", check], ["mediaPlaybackError", mediaFailed]]) {
          try { music.removeEventListener(name, callback); } catch (_) { /* Still settle. */ }
        }
        if (error) { pauseFailure(trace, error); reject(error); }
        else { trace.reason = "confirmed"; resolve({ accepted: true, playing: false, track_id: null, was_playing: wasPlaying }); }
      };
      const failed = (error) => finish(error || new PlaybackFailure("unknown"));
      const mediaFailed = () => finish(new PlaybackFailure("media_error"));
      const check = () => {
        if (settled) return;
        try {
          const error = playbackCurrentFailure(current, deadline, false, false);
          if (error) return failed(error);
          if (pauseActivity() === false && state.pendingStarts.size === 0) finish();
        } catch (error) { failed(error); }
      };
      try {
        const error = playbackCurrentFailure(current, deadline, false, false);
        if (error) return failed(error);
        music.addEventListener("playbackStateDidChange", check);
        music.addEventListener("mediaPlaybackError", mediaFailed);
        timer = window.setTimeout(() => failed(new PlaybackFailure("timeout")), Math.max(1, deadline - performance.now()));
        for (const pending of state.pendingStarts) pending.then(check, check);
        // Already quiet is a confirmed no-op. Pending native starts still
        // prevent early success, even if the SDK presently reports paused.
        check();
        if (settled) return;
        dispatchSDKPause(current, deadline, music).then(check, failed);
        check();
      } catch (error) { failed(error); }
    });
  }

  function pauseFailure(trace, error) {
    playbackFailure(trace, error);
    if (!authorized() && !["cancelled", "expired"].includes(trace.reason)) trace.reason = "authorization_lost";
  }

  async function executePause(command, sessionId, generation, wasPlaying, deadline, trace) {
    const music = state.music;
    const currentSession = () => state.sessionId === sessionId && state.generation === generation;
    // Public MusicKit.pause controls local media and does not validate Apple
    // authorization. A completed/paused player can remain quiet after sign-in
    // expires; starting music still requires authorization in the play path.
    const current = () => currentSession() && state.music === music;
    let outcome = { error: "playback_unconfirmed" };
    try {
      const error = playbackCurrentFailure(current, deadline, true, false);
      if (error) throw error;
      playbackPhase(trace, "confirm");
      outcome = await pauseAndConfirm(current, deadline, wasPlaying, trace);
    } catch (error) { pauseFailure(trace, error); }
    const diagnostics = playbackSnapshot(trace);
    // Authorization can age out without an SDK event. The same local bridge
    // still needs the failure result instead of waiting for its own timeout.
    if (!currentSession()) return;
    try {
      await api(`/v1/player/commands/${encodeURIComponent(command.id)}/result`, {
        method: "POST", body: { session_id: sessionId, result: outcome, diagnostics }, timeout: 300,
      });
    } catch (_) { /* No retry of an expired command; pause intent remains active. */ }
    renderPlayer();
  }

  function dispatchCommand(command, sessionId, generation) {
    // The server's remaining budget starts at delivery, including time spent
    // waiting behind an earlier command or an uncancellable SDK queue load.
    const received = performance.now();
    const trace = playbackTrace(received);
    const remaining = Number(command.expires_in_ms);
    if (command.operation === "pause") {
      const wasPlaying = state.playInFlight || state.manualStarting || state.pendingStarts.size > 0 ? true : pauseActivity();
      const deadline = received + Math.min(1200, remaining - 200);
      if (!Number.isFinite(remaining) || remaining <= 200) {
        executePause(command, sessionId, generation, wasPlaying, deadline, trace);
        return;
      }
      state.playEpoch += 1;
      state.commandAbort?.abort();
      state.manualAbort?.abort();
      state.pauseHold = true;
      state.queueReady = false;
      const pause = executePause(command, sessionId, generation, wasPlaying, deadline, trace);
      state.playChain = Promise.allSettled([state.playChain, pause]);
      return;
    }
    const deadline = received + Math.min(17500, remaining - 1500);
    const epoch = state.playEpoch;
    playbackPhase(trace, "waiting");
    state.playChain = state.playChain.catch(() => {}).then(async () => {
      if (epoch !== state.playEpoch || state.sessionId !== sessionId || state.generation !== generation) {
        playbackFailure(trace, new PlaybackFailure("cancelled"));
        playbackSnapshot(trace);
        return;
      }
      const abort = new AbortController();
      state.commandAbort = abort;
      state.playInFlight = true;
      try { await executeCommand(command, sessionId, generation, abort.signal, deadline, trace); }
      finally {
        if (state.commandAbort === abort) state.commandAbort = null;
        state.playInFlight = false;
      }
    });
  }

  function playAndConfirm(trackId, current, deadline, errorsBefore, signal, trace) {
    const music = state.music;
    return new Promise((resolve, reject) => {
      let settled = false;
      let invoked = false;
      let timer;
      const subscriptions = [];
      const finish = (error, fallback = "unknown") => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        for (const [name, callback] of subscriptions) {
          try { music.removeEventListener(name, callback); } catch (_) { /* Still clean remaining observers. */ }
        }
        signal?.removeEventListener("abort", cancelled);
        if (error) { playbackFailure(trace, error, fallback); reject(error); }
        else { trace.reason = "confirmed"; resolve({ accepted: true, playing: true, track_id: trackId }); }
      };
      const failed = (error) => finish(error || new PlaybackFailure("play_rejected"), "play_rejected");
      const cancelled = () => finish(playbackCurrentFailure(current, deadline) || new PlaybackFailure("cancelled"));
      const mediaFailed = () => finish(new PlaybackFailure("media_error"));
      const check = () => {
        if (settled) return;
        try {
          const error = playbackCurrentFailure(current, deadline);
          if (error) return finish(error);
          if (state.playbackErrors !== errorsBefore) return mediaFailed();
          if (invoked && playing() && observedTrackID() === trackId) finish();
        } catch (error) { failed(error); }
      };
      try {
        const error = playbackCurrentFailure(current, deadline);
        if (signal?.aborted || error) return finish(error || new PlaybackFailure("cancelled"));
        for (const name of ["playbackStateDidChange", "nowPlayingItemDidChange", "authorizationStatusDidChange"]) {
          music.addEventListener(name, check);
          subscriptions.push([name, check]);
        }
        music.addEventListener("mediaPlaybackError", mediaFailed);
        subscriptions.push(["mediaPlaybackError", mediaFailed]);
        signal?.addEventListener("abort", cancelled, { once: true });
        timer = window.setTimeout(cancelled, Math.max(1, deadline - performance.now()));
        // Observe before play: confirmation may precede settlement of the
        // native startup Promise. Always handle its rejection after settling.
        const finalError = playbackCurrentFailure(current, deadline);
        if (finalError) return finish(finalError);
        const operation = invokeSDKPlay(current, deadline, signal, () => {
          playbackPhase(trace, "play");
          state.pauseHold = false;
          invoked = true;
        }, () => { if (!settled) playbackPhase(trace, "confirm"); }, () => playbackPhase(trace, "waiting"));
        state.pendingStarts.add(operation);
        operation.then(() => { state.pendingStarts.delete(operation); check(); },
          (error) => { state.pendingStarts.delete(operation); failed(error); });
        check();
      } catch (error) { failed(error); }
    });
  }

  async function executeCommand(command, sessionId, generation, signal, deadline, trace) {
    const current = () => state.sessionId === sessionId && state.generation === generation && authorized() && !signal?.aborted;
    const started = performance.now();
    playbackPhase(trace, "preflight");
    let outcome = { error: "playback_unconfirmed" };
    try {
      const resuming = command.operation === "resume";
      const error = playbackCurrentFailure(current, deadline, true);
      if (error) throw error;
      if (!resuming && (command.operation !== "play" || !/^\d+$/.test(String(command.track_id)))) throw new PlaybackFailure("unknown");
      state.requestedTrack = resuming ? null : String(command.track_id);
      state.queueReady = false;
      ui.playerNote.textContent = resuming ? "Resuming your music…" : "Loading your requested song…";
      controls();
      const errorsBefore = state.playbackErrors;
      requireFullPlayback(state.music, window.MusicKit);
      // Cancelling our wait cannot cancel MusicKit's queue mutation. Serialize
      // the underlying operations so an old load cannot replace a newer queue.
      // Keep pause protection while any earlier native startup can still run.
      playbackPhase(trace, "waiting");
      await beforeDeadline(Promise.allSettled([...state.pendingStarts]), deadline, signal);
      await beforeDeadline(state.queueOperation, deadline, signal);
      const waitingError = playbackCurrentFailure(current, deadline);
      if (waitingError) throw waitingError;
      playbackPhase(trace, "queue");
      let trackId;
      if (resuming) {
        const queue = state.music.queue;
        const items = queue?.items;
        if (!queue || (Array.isArray(items) && items.length === 0)) {
          trace.reason = "queue_unavailable";
          trace.queue_check = "absent";
          outcome = { accepted: false, playing: false, track_id: null };
          ui.playerNote.textContent = "There is no song in the queue to resume.";
        } else {
          if (!Array.isArray(items)) { trace.queue_check = "wrong_length"; throw new PlaybackFailure("queue_unavailable"); }
          const item = queue.currentItem ?? (Number.isInteger(queue.position) ? items[queue.position] : null);
          trackId = item ? String(item.id) : observedTrackID();
          // Resume the existing current item, including an advanced queue.
          // A stale nowPlayingItem outside this queue is never resumable.
          const observed = observedTrackID();
          if (!/^[1-9][0-9]{0,19}$/.test(trackId || "") || !items.some((entry) => String(entry?.id) === trackId) ||
              (state.music.nowPlayingItem && observed !== trackId)) { trace.queue_check = "wrong_track"; throw new PlaybackFailure("queue_mismatch"); }
          trace.queue_check = "matched";
        }
      } else {
        trackId = String(command.track_id);
        const operation = Promise.resolve(state.music.setQueue({ song: trackId }));
        state.queueOperation = operation.catch(() => {});
        const queue = await beforeDeadline(operation, Math.min(deadline, started + 8000), signal);
        const queueError = playbackCurrentFailure(current, deadline);
        if (queueError) throw queueError;
        trace.queue_check = queueCheck(queue, trackId);
        // Filtered/unplayable songs must not resume an earlier queue.
        if (trace.queue_check !== "matched") throw new PlaybackFailure(trace.queue_check === "absent" ? "queue_unavailable" : "queue_mismatch");
      }
      if (trackId) {
        trace.trackId = trackId;
        state.requestedTrack = trackId;
        state.queueReady = true;
        ui.playerNote.textContent = "If the browser pauses here, click Start playback.";
        controls();
        outcome = await playAndConfirm(trackId, current, deadline, errorsBefore, signal, trace);
      }
    } catch (error) {
      // Do not retry a play operation whose outcome is unknown.
      playbackFailure(trace, error, trace.phase === "queue" ? "queue_rejected" : "unknown");
      ui.playerNote.textContent = playbackMessages[trace.reason] || playbackMessages.unknown;
    } finally {
      renderPlayer();
    }
    const diagnostics = playbackSnapshot(trace);
    if (state.sessionId !== sessionId || state.generation !== generation || signal?.aborted) return;
    try {
      await api(`/v1/player/commands/${encodeURIComponent(command.id)}/result`, {
        method: "POST", body: { session_id: sessionId, result: outcome, diagnostics }, timeout: 2000,
      });
    } catch (_) {
      ui.playerNote.textContent = "APM could not receive the playback result. This request will not be retried.";
    }
  }

  async function pollCommands(sessionId, generation) {
    const current = () => state.sessionId === sessionId && state.generation === generation;
    while (current()) {
      const abort = new AbortController();
      state.pollAbort = abort;
      try {
        const payload = await api(`/v1/player/commands?session_id=${encodeURIComponent(sessionId)}`, { signal: abort.signal, timeout: 22000 });
        if (!current()) return;
        if (ui.connection.dataset.state !== "connected") {
          connection("connected", "Apple Music connected", "Your catalog is ready. Keep this tab open for APM requests.");
        }
        const command = payload.command;
        if (command && typeof command.id === "string") {
          if (!state.processed.has(command.id)) {
            state.processed.add(command.id);
            if (state.processed.size > 256) state.processed.delete(state.processed.values().next().value);
            dispatchCommand(command, sessionId, generation);
          }
        }
      } catch (error) {
        if (!current() || error.name === "AbortError") return;
        if (error instanceof APIError && [400, 401, 403, 404, 409, 410].includes(error.status)) {
          loseConnection("The local player session ended. Connect Apple Music again to resume requests.");
          return;
        }
        connection("error", "Reconnecting to APM…", "Music can keep playing. Checking the local server again shortly.");
        await delay(1000);
      } finally {
        if (state.pollAbort === abort) state.pollAbort = null;
      }
    }
  }

  async function disconnect() {
    if (state.disconnecting) return;
    state.disconnecting = true;
    const sessionId = state.sessionId;
    stopPolling();
    state.sessionId = null;
    state.requestNumber += 1;
    state.requestBusy = false;
    controls();
    try {
      const music = state.music;
      const generation = state.generation;
      try { if (music) dispatchSDKPause(() => state.music === music && state.generation === generation && !state.sessionId,
        performance.now() + 2000, music).catch(() => {}); } catch (_) { /* Still revoke the session. */ }
      const revoke = sessionId ? api("/v1/player/disconnect", { method: "POST", body: { session_id: sessionId }, timeout: 3000 }) : Promise.resolve();
      const signOut = state.music ? beforeDeadline(Promise.resolve().then(() => state.music.unauthorize()), performance.now() + 5000) : Promise.resolve();
      const outcomes = await Promise.allSettled([revoke, signOut]);
      const incomplete = outcomes.some((outcome) => outcome.status === "rejected");
      connection(incomplete ? "error" : "ready", "Player disconnected", incomplete
        ? "The local player stopped. Apple Music sign-out could not be confirmed; close this tab if needed."
        : "Connect Apple Music whenever you want to listen again.");
    } finally {
      state.disconnecting = false;
      state.requestedTrack = null;
      state.queueReady = false;
      ui.candidates.replaceChildren();
      ui.candidates.hidden = true;
      ui.playerNote.textContent = "A browser may ask you to start playback with a click.";
      result("Connect Apple Music to get started.");
      renderPlayer();
    }
  }

  function showMusicResult(payload) {
    ui.candidates.replaceChildren();
    ui.candidates.hidden = true;
    const status = payload.status;
    if (status === "ambiguous" && Array.isArray(payload.candidates)) {
      result("Which recording would you like to play?");
      for (const candidate of payload.candidates) {
        if (typeof candidate.selection_id !== "string") continue;
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        button.className = "candidate";
        const title = document.createElement("span");
        title.className = "candidate-title";
        title.textContent = typeof candidate.title === "string" ? candidate.title : "Recording";
        const detail = document.createElement("span");
        detail.className = "candidate-detail";
        detail.textContent = [Array.isArray(candidate.artists) ? candidate.artists.join(", ") : "",
          candidate.album, candidate.version].filter((value) => typeof value === "string" && value).join(" · ");
        button.append(title, detail);
        button.addEventListener("click", () => requestMusic(`/v1/music/selections/${encodeURIComponent(candidate.selection_id)}/play`, {}));
        item.appendChild(button);
        ui.candidates.appendChild(item);
      }
      ui.candidates.hidden = ui.candidates.children.length === 0;
    } else if (status === "playing" && payload.playing === true) {
      result("Playback confirmed. Enjoy your music.", "success");
    } else if (status === "accepted") {
      result("Apple Music accepted the request. Playback has not yet been confirmed.");
    } else if (status === "not_found") {
      result("No matching recording was found. Try the song title with an artist.");
    } else if (status === "not_configured") {
      result("The Apple Music player is not connected. Connect above, then try again.");
    } else if (status === "unknown") {
      result("Playback could not be confirmed. Check the player, or click Start playback if the browser needs a gesture.", "error");
    } else {
      result(typeof payload.message === "string" ? payload.message : "The song could not be played. Check your Apple Music access and try another request.", "error");
    }
    controls();
  }

  async function requestMusic(path, body) {
    if (!state.sessionId || state.requestBusy) return;
    state.requestBusy = true;
    const requestNumber = ++state.requestNumber;
    result("Finding your recording and sending it to the player…");
    controls();
    try {
      const payload = await api(path, { method: "POST", body, timeout: 45000 });
      if (requestNumber === state.requestNumber) showMusicResult(payload);
    } catch (error) {
      if (requestNumber === state.requestNumber) result(error instanceof APIError ? error.message : "The song request could not be completed.", "error");
    } finally {
      if (requestNumber === state.requestNumber) state.requestBusy = false;
      controls();
    }
  }

  ui.start.addEventListener("click", () => {
    if (!authorized() || !state.requestedTrack || !state.queueReady ||
        !queueMatches(state.music.queue, state.requestedTrack) || state.manualStarting) return;
    state.manualStarting = true;
    const abort = new AbortController();
    state.manualAbort = abort;
    controls();
    try {
      requireFullPlayback(state.music, window.MusicKit);
      const epoch = state.playEpoch;
      const music = state.music;
      // This explicit gesture is a user's playback action, never an automatic
      // retry of a bridge command. Do not acknowledge an old command again.
      const operation = invokeSDKPlay(() => authorized() && state.music === music && state.playEpoch === epoch,
        performance.now() + 5000, abort.signal, () => { state.pauseHold = false; });
      state.pendingStarts.add(operation);
      operation.then(() => state.pendingStarts.delete(operation), () => state.pendingStarts.delete(operation));
      beforeDeadline(operation, performance.now() + 5000)
        .catch(() => { ui.playerNote.textContent = "Playback is still unavailable. Check your Apple Music subscription and browser audio settings."; })
        .finally(() => { if (state.manualAbort === abort) state.manualAbort = null; state.manualStarting = false; renderPlayer(); });
    } catch (_) {
      if (state.manualAbort === abort) state.manualAbort = null;
      state.manualStarting = false;
      ui.playerNote.textContent = "Apple Music could not start playback. Check your subscription and try again.";
      renderPlayer();
    }
  });

  ui.pause.addEventListener("click", () => {
    const failed = () => { ui.playerNote.textContent = "Playback could not be paused. Use the browser's audio controls."; };
    state.playEpoch += 1;
    state.commandAbort?.abort();
    state.manualAbort?.abort();
    state.pauseHold = true;
    const music = state.music;
    try { if (music) dispatchSDKPause(() => state.music === music && state.pauseHold, performance.now() + 2000, music).catch(failed); }
    catch (_) { failed(); }
    renderPlayer();
  });
  ui.connect.addEventListener("click", connect);
  ui.disconnect.addEventListener("click", disconnect);
  ui.reload.addEventListener("click", () => {
    if (initializing || state.connecting || state.disconnecting || state.sessionId || playing()) return;
    // Fetch the current HTML and script, not just a new token inside old code.
    // sessionStorage retains the local capability across this same-tab reload.
    window.location.reload();
  });
  ui.form.addEventListener("submit", (event) => {
    event.preventDefault();
    const title = ui.song.value.trim();
    const artist = ui.songArtist.value.trim();
    if (!title) { ui.song.focus(); return; }
    requestMusic("/v1/music/play", { title, ...(artist ? { artist } : {}) });
  });

  window.addEventListener("pagehide", () => {
    const sessionId = state.sessionId;
    stopPolling();
    state.sessionId = null;
    state.pauseHold = true;
    const music = state.music;
    const generation = state.generation;
    try { if (music) dispatchSDKPause(() => state.music === music && state.generation === generation && !state.sessionId,
      performance.now() + 1200, music).catch(() => {}); } catch (_) { /* Best effort during navigation. */ }
    if (sessionId && apiToken) {
      fetch("/v1/player/disconnect", { method: "POST", keepalive: true,
        headers: { Authorization: `Bearer ${apiToken}`, "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }) }).catch(() => {});
    }
  });

  initialize();
})();
