// Run with: node tests/test_music_browser.cjs
// Exercises the browser bridge without network access, credentials, or audio.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { webcrypto } = require("node:crypto");

const source = fs.readFileSync(path.join(__dirname, "../apm/static/music.js"), "utf8");
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const appleOrigin = "https://authorize.music.apple.com";
const diagnosticLabels = (text) => [...text.matchAll(/^\d+ms (authorization_started|authorization_resolved|authorization_rejected|local_registration_started|local_registration_succeeded|apple_(?:authorize|decline|unavailable|switch_user|close|third_party_info)|status_(?:-1|[0-3]))\b/gm)].map((match) => match[1]);
const throwingProperty = (name, object = {}) => Object.defineProperty(object, name, {
  get() { throw new Error("private-diagnostic-token"); },
});
const appleMessage = (method, extra = {}) => ({
  origin: appleOrigin,
  data: { jsonrpc: "2.0", method, params: { token: "private-diagnostic-token", url: "https://example.invalid/secret" } },
  ...extra,
});

const playbackEnums = {
  phase: ["preflight", "waiting", "queue", "play", "confirm"],
  reason: ["confirmed", "expired", "cancelled", "authorization_lost", "queue_rejected", "queue_unavailable", "queue_mismatch", "autoplay_blocked", "play_rejected", "media_error", "timeout", "unknown"],
  sdk_reason: ["ACCESS_DENIED", "AUTHORIZATION_ERROR", "CONTENT_EQUIVALENT", "CONTENT_RESTRICTED", "CONTENT_UNAVAILABLE", "CONTENT_UNSUPPORTED", "DEVICE_LIMIT", "GEO_BLOCK", "MEDIA_CERTIFICATE", "MEDIA_DESCRIPTOR", "MEDIA_LICENSE", "MEDIA_KEY", "MEDIA_PLAYBACK", "MEDIA_SESSION", "NETWORK_ERROR", "NOT_FOUND", "OUTPUT_RESTRICTED", "SERVER_ERROR", "SERVICE_UNAVAILABLE", "STREAM_UPSELL", "SUBSCRIPTION_ERROR", "TOKEN_EXPIRED", "UNAUTHORIZED_ERROR", "UNSUPPORTED_ERROR", "USER_INTERACTION_REQUIRED", "WIDEVINE_CDM_EXPIRED"],
  queue_check: ["absent", "different_queue", "wrong_length", "wrong_track", "matched"],
  state: ["none", "loading", "playing", "paused", "stopped", "ended", "seeking", "waiting", "stalled", "completed", "unknown"],
};
function validatePlaybackDiagnostic(diagnostic) {
  assert.ok(diagnostic && typeof diagnostic === "object");
  for (const field of ["phase", "reason", "elapsed_ms", "phase_ms"]) assert.ok(Object.hasOwn(diagnostic, field), field);
  for (const [field, value] of Object.entries(diagnostic)) {
    if (Object.hasOwn(playbackEnums, field)) assert.ok(playbackEnums[field].includes(value), `Unsafe ${field}: ${value}`);
    else if (["elapsed_ms", "phase_ms"].includes(field)) assert.ok(Number.isInteger(value) && value >= 0 && value <= 60000);
    else if (field === "track_matches") assert.ok(value === null || typeof value === "boolean");
    else assert.fail(`Unexpected diagnostic field: ${field}`);
  }
  assert.ok(diagnostic.phase_ms <= diagnostic.elapsed_ms);
  assert.doesNotMatch(JSON.stringify(diagnostic), /private-|test-api-capability|public-developer-jwt|https:\/\/|Example|Singer/);
}

class Element {
  constructor() {
    this.dataset = {}; this.listeners = {}; this.children = [];
    this.hidden = false; this.disabled = false; this.textContent = ""; this.value = "";
  }
  addEventListener(name, callback) { (this.listeners[name] ||= []).push(callback); }
  removeEventListener() {}
  querySelectorAll() { return this.children.filter((item) => item.tagName === "button"); }
  append(...items) { this.children.push(...items); }
  appendChild(item) { this.children.push(item); }
  replaceChildren(...items) { this.children = items; }
  removeAttribute(name) { delete this[name]; }
  focus() {}
}

async function scenario(options = {}) {
  const elements = new Map();
  const events = new Map();
  const sdkEvents = new Map();
  const listen = (registry, name, callback) => {
    if (!registry.has(name)) registry.set(name, new Set());
    registry.get(name).add(callback);
  };
  const unlisten = (registry, name, callback) => registry.get(name)?.delete(callback);
  const emit = (registry, name, event) => {
    for (const callback of [...(registry.get(name) || [])]) callback(event);
  };
  const calls = [];
  const storage = new Map();
  if (options.cachedToken) storage.set("apm.apple_music.api_token", "test-api-capability");
  const get = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  let configured = 0;
  let reloads = 0;
  const configuredTokens = [];
  let authorizations = 0;
  let plays = 0;
  let pauses = 0;
  let queues = 0;
  let dispatched = 0;
  let emptyPollSent = false;
  let connectionLossSent = false;
  let settlePlay;
  let queueLoads = 0;
  let maxQueueLoads = 0;
  const completedQueues = [];
  const playedTracks = [];
  const location = { hash: options.noToken || options.cachedToken ? "" : "#token=test-api-capability", pathname: "/",
    search: options.diagnostics ? "?diagnostics=1" : (options.diagnosticsSearch || ""),
    reload() { reloads += 1; } };
  const mediaItem = (id) => ({ id, attributes: { name: "Example", artistName: "Singer" } });
  const resumeFixture = options.resumeCommand || options.pauseAfterResume || options.controlSequence;
  const initialItems = options.resumeEmpty ? [] : [mediaItem("111"), ...(options.resumeAdvanced ? [mediaItem("222")] : [])];
  const initialQueue = { items: initialItems, position: options.resumeNoCurrent ? -1 : options.resumeAdvanced ? 1 : 0,
    get currentItem() { return this.items[this.position]; } };
  const music = {
    isAuthorized: false, storefrontId: options.invalidRegion ? "private-region-value" : "us", playbackMode: 0,
    playbackState: options.sequencePlaying ? 2 : resumeFixture ? 3 : options.pauseCommand ? (options.pauseCompleted ? 10 : options.pauseStateUnknown ? 99 : options.pauseIdle ? 3 : options.pauseLoading ? 1 : 2) : 0,
    nowPlayingItem: resumeFixture ? (options.resumeStale ? mediaItem(options.resumeInvalidID ? "i.library" : "999") : initialQueue.currentItem) : null,
    queue: options.resumeAbsent ? null : initialQueue,
    currentPlaybackTime: 47.25,
    authorizationStatus: 0,
    addEventListener(name, callback) { listen(sdkEvents, name, callback); },
    removeEventListener(name, callback) { unlisten(sdkEvents, name, callback); },
    async authorize() {
      authorizations += 1;
      if (options.reloadGuards) {
        get("reload-setup").listeners.click[0]();
        assert.equal(reloads, 0, "A pending authorization must not reload the page");
      }
      for (const event of options.authEvents || []) {
        if (event.message) emit(events, "message", event.message);
        if (event.statusEvent) emit(sdkEvents, "authorizationStatusDidChange", event.statusEvent);
        if (Object.hasOwn(event, "status")) {
          this.authorizationStatus = event.status;
          emit(sdkEvents, "authorizationStatusDidChange", { authorizationStatus: event.status });
        }
      }
      if (options.authError) throw options.authError;
      this.isAuthorized = !options.cancel;
    },
    async unauthorize() { this.isAuthorized = false; },
    async setQueue({ song }) {
      queues += 1;
      if (options.queueSDKError) throw options.queueSDKError;
      queueLoads += 1;
      maxQueueLoads = Math.max(maxQueueLoads, queueLoads);
      if (options.firstQueueDelay && queues === 1) await sleep(options.firstQueueDelay);
      if (options.queueDelay) await sleep(options.queueDelay);
      queueLoads -= 1;
      if (options.noQueue) return undefined;
      const queue = { items: options.emptyQueue ? [] : [mediaItem(options.wrongQueue ? "456" : song)] };
      if (!options.staleQueue) this.queue = queue;
      completedQueues.push(song);
      return queue;
    },
    play() {
      plays += 1;
      playedTracks.push((this.queue.currentItem || this.queue.items[0])?.id);
      if (options.playError && (plays === 1 || !options.manualRetry)) throw options.playSDKError || new Error("private-sdk-detail");
      if (options.rejectPlay) return Promise.reject(options.playSDKError || new Error("private-sdk-detail"));
      const publishPlayback = () => {
        this.nowPlayingItem = options.wrongPlayingTrack ? mediaItem("456") : (this.queue.currentItem || this.queue.items[0]);
        this.playbackState = 2;
        emit(sdkEvents, "playbackStateDidChange", { state: 2 });
      };
      if (options.race === "manual_start" && plays === 2) {
        return new Promise((resolve) => setTimeout(() => { publishPlayback(); resolve(); }, 80));
      }
      if (options.authLostDuringPlayback) setTimeout(() => {
        this.isAuthorized = false;
        emit(sdkEvents, "authorizationStatusDidChange", { authorizationStatus: 0 });
      }, 5);
      else if (options.abortDuringPlayback) setTimeout(() => emit(events, "pagehide", {}), 5);
      else if (options.mediaErrorEvent) setTimeout(() => emit(sdkEvents, "mediaPlaybackError", {}), 5);
      else if (options.playEventDelay) setTimeout(publishPlayback, options.playEventDelay);
      else publishPlayback();
      if (options.correctTrackLater) setTimeout(() => {
        this.nowPlayingItem = this.queue.items[0];
        emit(sdkEvents, "nowPlayingItemDidChange", { item: this.nowPlayingItem });
      }, 5);
      if (options.pendingStartDelay) return sleep(options.pendingStartDelay);
      if (options.deferredPlay || options.mediaErrorEvent || options.abortDuringPlayback || options.authLostDuringPlayback) {
        return new Promise((resolve, reject) => { settlePlay = { resolve, reject }; });
      }
    },
    pause() {
      pauses += 1;
      if (options.pauseThrow) throw new Error("private-sdk-detail");
      if (options.pauseReject) return Promise.reject(new Error("private-sdk-detail"));
      const quiet = () => {
        if (!options.pauseNoEffect) { this.playbackState = 3; emit(sdkEvents, "playbackStateDidChange", { state: 3 }); }
      };
      if (options.pauseDelay) return sleep(options.pauseDelay).then(quiet);
      quiet();
    },
  };
  const sdkInvocations = { play: [], pause: [] };
  let suppressedControls = 0;
  if (options.controlSequence) {
    // Mirrors the official SDK's immediate AsyncDebounce: every public call
    // extends its per-method window, even if the previous Promise settled.
    for (const method of ["play", "pause"]) {
      const original = music[method].bind(music);
      let next = -Infinity;
      music[method] = (...args) => {
        const now = Date.now();
        const suppressed = now < next;
        next = now + 250;
        if (suppressed) { suppressedControls += 1; return Promise.resolve(); }
        sdkInvocations[method].push(now);
        return Promise.resolve(original(...args));
      };
    }
  }
  if (options.noFullMode) delete music.playbackMode;
  const MusicKit = {
    PlaybackMode: { FULL_PLAYBACK_ONLY: 2 },
    PlaybackStates: { none: 0, loading: 1, playing: 2, paused: 3, stopped: 4, ended: 5, seeking: 6, waiting: 8, stalled: 9, completed: 10 },
    async configure(config) {
      configured += 1;
      configuredTokens.push(config.developerToken);
      assert.equal(location.hash, "");
      return music;
    },
  };
  const window = {
    location, MusicKit,
    setTimeout(callback, milliseconds) {
      // Simulate an early fractional timer callback without advancing the true
      // performance clock. Deadline observers must recheck their budget.
      const actual = options.earlyTimers && milliseconds > 2 && milliseconds < 100 ? milliseconds - 2 : milliseconds;
      const timer = setTimeout(callback, actual); timer.unref(); return timer;
    },
    clearTimeout,
    addEventListener(name, callback) { listen(events, name, callback); },
    removeEventListener(name, callback) { unlisten(events, name, callback); },
  };
  const document = {
    getElementById: get,
    createElement(tag) { const element = new Element(); element.tagName = tag; return element; },
    head: new Element(), addEventListener() {}, removeEventListener() {},
  };
  const fetch = async (url, init = {}) => {
    calls.push({ url, init });
    assert.equal(init.headers.Authorization, "Bearer test-api-capability");
    const response = (body) => ({ ok: true, status: 200, json: async () => body });
    if (url === "/v1/config") {
      if (options.reloadGuards) {
        get("reload-setup").listeners.click[0]();
        assert.equal(reloads, 0, "Configuration in progress must not reload the page");
      }
      return response(options.unconfigured
        ? { configured: false, message: "Apple developer key is not configured." }
        : { configured: true, developer_token: "public-developer-jwt", app_name: "APM Assistant" });
    }
    if (url === "/v1/player/session") {
      assert.equal(JSON.parse(init.body).protocol_version, 3, "The player must advertise confirmed pause and resume support");
      return options.sessionError
        ? { ok: false, status: options.sessionError, json: async () => ({ detail: "private-server-detail" }) }
        : response({ ok: true });
    }
    if (url.startsWith("/v1/player/commands?")) {
      if (options.connectionLoss && dispatched >= (options.connectionLossAfter || 1) && !connectionLossSent) {
        connectionLossSent = true;
        await sleep(options.lossDuringQueue || options.lossDuringStart ? 5 : 15);
        if (options.connectionLoss === "network") throw new TypeError("Synthetic offline transport");
        return { ok: false, status: options.connectionLoss, json: async () => ({ error: "Synthetic session failure" }) };
      }
      if (options.reconnectAfterLoss && connectionLossSent && dispatched === 1 &&
          calls.filter((call) => call.url === "/v1/player/session").length === 2) {
        dispatched += 1;
        return response({ command: { id: "new-resume", operation: "resume", expires_in_ms: 20000 } });
      }
      if (options.commands && dispatched < options.commands.length) {
        const { deliveryDelay = 0, afterResult, manualPause, ...command } = options.commands[dispatched++];
        if (afterResult) {
          const until = performance.now() + 1500;
          while (!calls.some((call) => call.url.endsWith(`/${afterResult}/result`))) {
            assert.ok(performance.now() < until, "Prior command must complete without a retry");
            await sleep(2);
          }
        }
        if (deliveryDelay) await sleep(deliveryDelay);
        if (manualPause) {
          get("pause-playback").listeners.click[0]();
          await sleep(10);
          assert.equal(music.playbackState, 3, "The manual pause must be observed before the wake command");
        }
        if (options.race === "manual_start" && command.operation === "pause") {
          music.playbackState = 3;
          emit(sdkEvents, "playbackStateDidChange", { state: 3 });
          get("start-playback").listeners.click[0]();
        }
        return response({ command });
      }
      if ((options.pauseCommand && dispatched === 0) || ((options.pauseAfterPlay || options.pauseAfterResume) && dispatched === 1)) {
        dispatched += 1;
        if (options.pauseAfterPlay || options.pauseAfterResume) await sleep(5);
        if (options.pauseAuthLostSilently) music.isAuthorized = false;
        return response({ command: { id: "pause-command", operation: "pause",
          expires_in_ms: options.pauseShort ? 260 : options.pauseExpired ? 0 : 1500 } });
      }
      if (options.emptyPollFirst && !emptyPollSent) {
        emptyPollSent = true;
        return response({ command: null });
      }
      if (!options.commands && dispatched < (options.duplicate ? 2 : 1)) {
        dispatched += 1;
        return response({ command: { id: "test-command", operation: resumeFixture ? "resume" : "play", ...(resumeFixture ? {} : { track_id: "123" }),
          expires_in_ms: options.expired ? 0 : (options.queueDelay || options.shortDeadline ? 1550 : 20000) } });
      }
      return new Promise((resolve, reject) => {
        init.signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")), { once: true });
      });
    }
    return response({ ok: true });
  };
  // Mirrors the HTML's initial hidden state without requiring a DOM package.
  get("auth-diagnostics").hidden = true;
  get("playback-diagnostics").hidden = true;
  vm.runInNewContext(source, {
    window, document, history: { replaceState() { location.hash = ""; } },
    sessionStorage: { setItem(key, value) { storage.set(key, value); }, getItem(key) { return storage.get(key); } },
    URLSearchParams, URL, performance, crypto: webcrypto, AbortController, DOMException, fetch,
  });
  await sleep(20);
  assert.equal(location.hash, "");
  if (options.noToken) {
    assert.equal(calls.length, 0);
    assert.equal(configured, 0);
    assert.equal(get("connect").disabled, true);
    assert.match(get("connection-detail").textContent, /Open the player link/);
    return;
  }
  assert.equal(storage.get("apm.apple_music.api_token"), "test-api-capability");
  if (options.unconfigured) {
    assert.equal(configured, 0);
    assert.equal(get("connection-title").textContent, "Apple Music needs setup");
    assert.equal(get("connect").disabled, true);
    assert.equal(get("reload-setup").hidden, false);
    return;
  }
  if (options.noFullMode) {
    assert.equal(get("connect").disabled, true);
    assert.match(get("connection-detail").textContent, /cannot enforce full-song playback/);
    return;
  }
  assert.equal(music.playbackMode, 2);
  if (options.expectedDiagnostics) {
    const before = get("auth-diagnostics-output").textContent;
    emit(events, "message", appleMessage("close"));
    emit(sdkEvents, "authorizationStatusDidChange", { authorizationStatus: -1 });
    assert.equal(get("auth-diagnostics-output").textContent, before, "No trace is captured before an explicit authorization attempt");
  }
  let subscriptionsBeforePlayback = [...sdkEvents].map(([name, callbacks]) => [name, callbacks.size]);
  await get("connect").listeners.click[0]();
  await sleep(options.queueDelay ? options.queueDelay + 40 : options.shortDeadline || options.pauseShort ? 120 : 30);
  if (options.rapidReconnect) {
    const limit = performance.now() + 100;
    while (get("connection-title").textContent !== "Player disconnected") {
      assert.ok(performance.now() < limit, "The connection must fail before reconnecting");
      await sleep(1);
    }
    assert.equal(pauses, 1, "The new safety pause must still be waiting on the SDK cooldown");
    assert.equal(music.playbackState, 2);
    await get("connect").listeners.click[0]();
  }
  if (options.manualDisconnect) await Promise.race([
    get("disconnect").listeners.click[0](),
    sleep(1500).then(() => { throw new Error("Disconnect did not finish within its local pause budget"); }),
  ]);
  if (options.connectionLoss || options.manualDisconnect) {
    await sleep(options.pauseNoEffect ? 1250 : 400);
    assert.equal(get("connection-title").textContent, options.rapidReconnect ? "Apple Music connected" : "Player disconnected");
    assert.equal(get("start-playback").disabled, true);
    assert.equal(calls.filter((call) => call.url.startsWith("/v1/player/commands?")).length, options.rapidReconnect ? 4 : 2,
      "A failed connection must not retry polling or recover playback automatically");
    const revocations = calls.filter((call) => call.url === "/v1/player/disconnect");
    assert.equal(revocations.length, 1);
    assert.equal(JSON.parse(revocations[0].init.body).session_id,
      JSON.parse(calls.find((call) => call.url === "/v1/player/session").init.body).session_id);
    if (options.pauseNoEffect) {
      assert.equal(music.playbackState, 2);
      assert.match(get("player-note").textContent, /pause could not be confirmed/);
      assert.doesNotMatch(get("player-note").textContent, /Music is paused/);
    } else {
      assert.notEqual(music.playbackState, 2, "Lost sessions and late starts must not leave audio playing");
      if (!options.rapidReconnect) assert.match(get("player-note").textContent, /Music is paused/);
      if (!options.lossDuringQueue) assert.ok(pauses >= 1);
    }
    assert.equal(plays, options.lossDuringQueue ? 0 : 1, "No playback is retried after connection loss");
    if (options.rapidReconnect) {
      assert.equal(pauses, 2, "Reconnection must preserve the deferred safety pause");
      assert.equal(suppressedControls, 0);
      assert.ok(sdkInvocations.pause[1] - sdkInvocations.pause[0] >= 250);
      assert.doesNotMatch(get("player-note").textContent, /Reconnect the player|Connection ended/,
        "The old connection must not overwrite the replacement session's status");
    }
    if (options.reconnectAfterLoss) {
      await get("connect").listeners.click[0]();
      await sleep(300);
      assert.equal(get("connection-title").textContent, "Apple Music connected");
      assert.equal(music.playbackState, 2, "An explicit new resume can release the old pause hold");
      assert.equal(plays, 2);
      assert.equal(queues, 1, "Resume must preserve the existing queue");
      assert.equal(calls.filter((call) => call.url === "/v1/player/disconnect").length, 1,
        "The previous loss must not revoke the replacement session");
    }
    emit(events, "pagehide", {});
    return;
  }
  if (options.authLostDuringPlayback) await sleep(1250);
  if (options.race === "manual_start") {
    assert.equal(calls.some((call) => call.url.includes("pause-command/result")), false,
      "Pause cannot confirm quiet while a manual native play Promise may still start audio");
  }
  if (options.race) await sleep(350 + (options.commands || []).reduce((total, command) => total + (command.deliveryDelay || 0), 0));
  if (options.pauseAfterResume || options.pauseAfterPlay) await sleep(300);
  if (options.controlSequence) await sleep(650);
  if (options.expectedDiagnostics) {
    const trace = get("auth-diagnostics-output").textContent;
    assert.equal(get("auth-diagnostics").hidden, !options.diagnostics);
    assert.deepEqual(diagnosticLabels(trace), options.expectedDiagnostics);
    assert.ok((trace.match(/^\d+ms /gm) || []).length <= 40, "Diagnostic event lines must be bounded");
    if (options.diagnosticFlood) {
      assert.match(trace, /^\d+ms authorization_started$/m);
      assert.match(trace, /^\d+ms final_status_0$/m);
      assert.match(trace, /Apple reported authorization unavailable/);
    }
    for (const secret of ["private-diagnostic-token", "private-sdk-token", "https://example.invalid/secret", "test-api-capability", "public-developer-jwt"]) {
      assert.equal(trace.includes(secret), false, `Diagnostics must exclude ${secret}`);
    }
    emit(events, "message", appleMessage("unavailable"));
    emit(sdkEvents, "authorizationStatusDidChange", { authorizationStatus: 0 });
    assert.equal(get("auth-diagnostics-output").textContent, trace, "The finished trace must not absorb later messages or cleanup status events");
    assert.equal(authorizations, 1, "Diagnostics must not initiate a second authorization attempt");
  }
  if (options.authError || options.invalidRegion || options.sessionError) {
    const title = get("connection-title").textContent;
    const detail = get("connection-detail").textContent;
    assert.equal(plays, 0);
    assert.equal(get("reload-setup").hidden, false);
    assert.equal(detail.includes("private"), false);
    if (options.authError) {
      assert.equal(title, "Apple Music sign-in failed");
      assert.equal(calls.filter((call) => call.url === "/v1/player/session").length, 0);
      if (options.authError.reason === "UNAUTHORIZED_ERROR") {
        assert.match(detail, /UNAUTHORIZED_ERROR; HTTP 401/);
        assert.match(detail, /Reload player/);
        assert.equal(detail.includes("subscription"), false);
      } else {
        assert.equal(detail.includes("HTTP"), false);
        assert.equal(detail.includes("TOKEN_SECRET"), false);
      }
    } else if (options.invalidRegion) {
      assert.equal(title, "Apple Music region unavailable");
      assert.match(detail, /sign-in succeeded/);
      assert.equal(calls.filter((call) => call.url === "/v1/player/session").length, 0);
    } else {
      assert.equal(title, "APM player connection failed");
      assert.match(detail, /Local API: HTTP 503/);
      assert.match(detail, /local APM server/);
    }
    if (!options.reloadAfterError) return;
    await get("reload-setup").listeners.click[0]();
    assert.equal(reloads, 1, "Reload player must perform a full document reload");
    assert.equal(configured, 1, "Old script must not just reconfigure the SDK");
    assert.deepEqual(configuredTokens, ["public-developer-jwt"]);
    assert.equal(authorizations, 1, "Reload must not reopen authorization in old code");
    assert.equal(storage.get("apm.apple_music.api_token"), "test-api-capability");
    assert.equal(location.hash, "", "Reload must not reintroduce credentials into the URL");
    return;
  }
  if (options.cancel) {
    assert.equal(calls.filter((call) => call.url === "/v1/player/session").length, 0);
    assert.equal(plays, 0);
    return;
  }
  const acknowledgements = calls.filter((call) => call.url.endsWith("/result"));
  for (const call of acknowledgements) validatePlaybackDiagnostic(JSON.parse(call.init.body).diagnostics);
  const localDiagnostics = get("playback-diagnostics-output").textContent;
  if (localDiagnostics) {
    assert.equal(get("playback-diagnostics").hidden, false);
    const diagnostic = JSON.parse(localDiagnostics);
    validatePlaybackDiagnostic(diagnostic);
    if (options.expectedPlaybackDiagnostic) {
      for (const [key, value] of Object.entries(options.expectedPlaybackDiagnostic)) assert.equal(diagnostic[key], value, `Diagnostic ${key}`);
    }
    if (options.earlyTimers) assert.ok(diagnostic.elapsed_ms >= 50, "Early timers must preserve the absolute command budget");
    if (options.noPlaybackSDKReason) assert.equal(Object.hasOwn(diagnostic, "sdk_reason"), false);
  }
  if (options.expectedPlaybackDiagnostic) assert.ok(localDiagnostics, "The latest diagnostic must remain visible even if a session ended");
  assert.doesNotMatch(get("player-note").textContent, /private-|test-api-capability|public-developer-jwt/);
  assert.deepEqual([...sdkEvents].map(([name, callbacks]) => [name, callbacks.size]), subscriptionsBeforePlayback,
    "Every command observer must be removed after confirmation, rejection, timeout, or cancellation");
  if (options.controlSequence) {
    const outcomes = Object.fromEntries(acknowledgements.map((call) => [call.url.split("/").at(-2), JSON.parse(call.init.body)]));
    assert.equal(suppressedControls, 0, "No public SDK call should fall into its suppression window");
    assert.equal(queues, 0);
    assert.equal(music.currentPlaybackTime, 47.25);
    assert.equal(music.queue, initialQueue);
    if (options.controlSequence === "pause_resume_pause") {
      assert.equal(acknowledgements.length, 3);
      assert.equal(outcomes.first.result.playing, false);
      assert.equal(outcomes.middle.result.playing, true);
      assert.equal(outcomes.last.result.playing, false);
      assert.equal(plays, 1);
      assert.equal(pauses, 2);
      assert.ok(sdkInvocations.pause[1] - sdkInvocations.pause[0] >= 250);
      assert.ok(outcomes.last.diagnostics.elapsed_ms < 1000, "Pause keeps its original short deadline");
    } else if (options.controlSequence === "resume_pause_resume") {
      assert.equal(acknowledgements.length, 3);
      assert.equal(outcomes.last.result.playing, true);
      assert.equal(plays, 2);
      assert.equal(pauses, 1);
      assert.ok(sdkInvocations.play[1] - sdkInvocations.play[0] >= 250);
    } else if (options.controlSequence === "cancel_gated_resume") {
      assert.equal(acknowledgements.length, 3);
      assert.equal(outcomes.gated, undefined, "A preempted delayed resume must not acknowledge or invoke play");
      assert.equal(outcomes.last.result.playing, false);
      assert.equal(plays, 1);
      assert.equal(music.playbackState, 3);
    } else if (options.controlSequence === "expire_gated_resume") {
      assert.equal(acknowledgements.length, 3);
      assert.deepEqual(outcomes.last.result, { error: "playback_unconfirmed" });
      assert.equal(outcomes.last.diagnostics.reason, "timeout");
      assert.equal(outcomes.last.diagnostics.phase, "waiting");
      assert.equal(plays, 1);
      assert.equal(music.playbackState, 3);
    } else if (options.controlSequence === "manual_paused_wake") {
      assert.equal(acknowledgements.length, 2);
      assert.deepEqual(outcomes.last.result, { accepted: true, playing: false, track_id: null, was_playing: false });
      assert.equal(outcomes.last.diagnostics.reason, "confirmed");
      assert.equal(outcomes.last.diagnostics.state, "paused");
      assert.equal(plays, 1);
      assert.equal(pauses, 1, "An already quiet wake must not invoke another native pause");
    } else if (options.controlSequence === "manual_paused_pending_start") {
      assert.equal(acknowledgements.length, 2);
      assert.deepEqual(outcomes.last.result, { error: "playback_unconfirmed" });
      assert.equal(outcomes.last.diagnostics.reason, "timeout");
      assert.equal(outcomes.last.diagnostics.state, "paused");
      assert.equal(plays, 1);
      assert.equal(pauses, 1);
      assert.ok(settlePlay, "This failure needs a still-unresolved native play, not merely a manual pause");
    } else if (options.controlSequence === "coalesced_pause") {
      assert.equal(acknowledgements.length, 2);
      assert.equal(outcomes.first.result.playing, false);
      assert.equal(outcomes.last.result.playing, false);
      assert.equal(pauses, 1, "Concurrent pauses must share one native invocation");
      assert.equal(plays, 0);
    }
    if (settlePlay) { settlePlay.reject(new Error("private-sdk-detail")); await sleep(0); }
    emit(events, "pagehide", {});
    return;
  }
  if (options.race) {
    const results = Object.fromEntries(acknowledgements.map((call) => [call.url.split("/").at(-2), JSON.parse(call.init.body).result]));
    assert.equal(acknowledgements.length, 2);
    if (options.race === "manual_start") {
      assert.deepEqual(results["pause-command"], { accepted: true, playing: false, track_id: null, was_playing: true });
      assert.equal(plays, 2);
      assert.equal(music.playbackState, 3);
      assert.equal(pauses, 2, "A late manual start remains covered by the pause intent");
    } else if (options.race === "queue_replacement") {
      assert.equal(results["pause-command"].playing, false);
      assert.equal(results[options.race === "resume_paused_expiry" || options.race === "queued_resume_expiry" ? "second-resume" : "second-play"].track_id, "456");
      assert.equal(maxQueueLoads, 1, "Underlying queue mutations must remain serialized after cancellation");
      assert.deepEqual(completedQueues, ["123", "456"]);
      assert.deepEqual(playedTracks, ["456"]);
      assert.equal(music.queue.items[0].id, "456", "The late old queue must not replace the new request");
    } else if (options.race === "paused_expiry" || options.race === "resume_paused_expiry") {
      assert.deepEqual(results["pause-command"], { error: "playback_unconfirmed" });
      assert.deepEqual(results[options.race === "resume_paused_expiry" || options.race === "queued_resume_expiry" ? "second-resume" : "second-play"], { error: "playback_unconfirmed" });
      assert.deepEqual(completedQueues, ["123"]);
      assert.deepEqual(playedTracks, ["123"]);
      assert.equal(music.playbackState, 3, "An expired request must not release the pause hold protecting against a late native start");
      assert.equal(pauses, 2);
    } else {
      assert.equal(results["test-command"].track_id, "123");
      assert.deepEqual(results[options.race === "resume_paused_expiry" || options.race === "queued_resume_expiry" ? "second-resume" : "second-play"], { error: "playback_unconfirmed" });
      assert.deepEqual(completedQueues, ["123"], "A command that expired in the queue must not mutate MusicKit");
      assert.deepEqual(playedTracks, ["123"]);
    }
    emit(events, "pagehide", {});
    return;
  }
  if (resumeFixture) {
    assert.equal(queues, 0, "Resume must never replace the MusicKit queue");
    assert.equal(music.queue, options.resumeAbsent ? null : initialQueue);
    assert.equal(music.currentPlaybackTime, 47.25, "Resume must preserve the paused position");
    if (options.pauseAfterResume) {
      assert.equal(acknowledgements.length, 1, "Pause must cancel the resume observer and acknowledgement");
      assert.ok(acknowledgements[0].url.includes("pause-command"));
      assert.deepEqual(JSON.parse(acknowledgements[0].init.body).result,
        { accepted: true, playing: false, track_id: null, was_playing: true });
      assert.equal(plays, 1);
      assert.equal(music.playbackState, 3, "A late native resume must remain covered by pause");
    } else if (options.abortDuringPlayback) {
      assert.equal(acknowledgements.length, 0, "A disconnected session must not receive late resume confirmation");
      assert.equal(plays, 1);
    } else {
      assert.equal(acknowledgements.length, 1);
      const response = JSON.parse(acknowledgements[0].init.body).result;
      const empty = options.resumeEmpty || options.resumeAbsent;
      const blocked = options.expired || (!empty && (options.resumeStale || options.resumeNoCurrent));
      assert.equal(plays, empty || blocked ? 0 : 1, "Resume must not retry or start an unrelated track");
      if (options.expired || blocked || options.playError || options.rejectPlay || options.wrongPlayingTrack || options.latePlaying) {
        assert.deepEqual(response, { error: "playback_unconfirmed" });
      } else if (empty) {
        assert.deepEqual(response, { accepted: false, playing: false, track_id: null });
      } else {
        assert.deepEqual(response, { accepted: true, playing: true, track_id: options.resumeAdvanced ? "222" : "111" });
        assert.deepEqual(playedTracks, [options.resumeAdvanced ? "222" : "111"]);
      }
    }
    if (settlePlay) { settlePlay.reject(new Error("private-sdk-detail")); await sleep(0); }
    emit(events, "pagehide", {});
    return;
  }
  if (options.pauseCommand || options.pauseAfterPlay) {
    assert.equal(acknowledgements.length, 1, "A preempted play must not acknowledge after the pause");
    assert.ok(acknowledgements[0].url.includes("pause-command"));
    const result = JSON.parse(acknowledgements[0].init.body).result;
    if (options.pauseAuthLostSilently) {
      const quietBefore = options.pauseIdle || options.pauseCompleted;
      const cannotConfirm = options.pauseNoEffect || options.pauseReject || options.deferredPlay;
      if (cannotConfirm) {
        assert.deepEqual(result, { error: "playback_unconfirmed" });
        assert.equal(JSON.parse(acknowledgements[0].init.body).diagnostics.reason, "authorization_lost");
      } else {
        assert.deepEqual(result, { accepted: true, playing: false, track_id: null, was_playing: !quietBefore });
        assert.equal(JSON.parse(acknowledgements[0].init.body).diagnostics.reason, "confirmed");
      }
      if (quietBefore) assert.equal(pauses, 0, "Known silence needs no native call or Apple authorization");
      else assert.ok(pauses >= 1, "Active or unknown media still needs an attempted and observed pause");
      assert.equal(plays, options.pauseAfterPlay ? 1 : 0);
      if (settlePlay) { settlePlay.reject(new Error("private-sdk-detail")); await sleep(0); }
      emit(events, "pagehide", {});
      return;
    }
    const unconfirmed = (!options.pauseIdle && (options.pauseReject || options.pauseThrow || options.pauseNoEffect)) || options.pauseExpired || options.deferredPlay;
    if (unconfirmed) assert.deepEqual(result, { error: "playback_unconfirmed" });
    else assert.deepEqual(result, { accepted: true, playing: false, track_id: null,
      was_playing: !options.pauseIdle });
    assert.equal(plays, options.pauseAfterPlay && !options.queueDelay ? 1 : 0);
    if (!unconfirmed && !options.queueDelay) assert.equal(music.playbackState, 3);
    assert.equal(pauses, options.pauseExpired || options.pauseIdle || (options.pauseAfterPlay && options.queueDelay) ? 0 : options.deferredPlay ? 2 : 1);
    if (settlePlay) { settlePlay.reject(new Error("private-sdk-detail")); await sleep(0); }
    emit(events, "pagehide", {});
    return;
  }
  if (options.abortDuringPlayback || options.authLostDuringPlayback) {
    assert.equal(acknowledgements.length, 0, "A cancelled session must not receive a late acknowledgement");
    assert.equal(plays, 1);
    settlePlay.reject(new Error("private-sdk-detail"));
    await sleep(0);
    return;
  }
  assert.equal(acknowledgements.length, 1);
  const result = JSON.parse(acknowledgements[0].init.body).result;
  const blocked = options.emptyQueue || options.wrongQueue || options.staleQueue || options.noQueue || options.expired || options.queueDelay || options.queueSDKError;
  assert.equal(queues, options.expired ? 0 : 1);
  assert.equal(plays, blocked ? 0 : 1);
  if (blocked || options.playError || options.rejectPlay || options.mediaErrorEvent ||
      (options.wrongPlayingTrack && !options.correctTrackLater) || options.latePlaying) {
    assert.deepEqual(result, { error: "playback_unconfirmed" });
  }
  else assert.deepEqual(result, { accepted: true, playing: true, track_id: "123" });
  if (settlePlay) {
    // Confirmation/rejection must finish while play() is still pending. A late
    // rejection must be handled without another acknowledgement or play call.
    settlePlay.reject(new Error("private-sdk-detail"));
    await sleep(0);
    assert.equal(calls.filter((call) => call.url.endsWith("/result")).length, 1);
    assert.equal(plays, 1);
  }
  assert.equal(get("player-note").textContent.includes("private-sdk-detail"), false);
  if (options.manualRetry) {
    assert.equal(get("start-playback").disabled, false);
    get("start-playback").listeners.click[0]();
    await sleep(280);
    assert.equal(plays, 2);
    assert.equal(calls.filter((call) => call.url.endsWith("/result")).length, 1);
    assert.equal(get("playback-state").textContent, "Playing");
  }
  assert.equal(get("reload-setup").hidden, true);
  const configurationsBefore = configured;
  await get("reload-setup").listeners.click[0]();
  assert.equal(configured, configurationsBefore, "Connected playback must not be reconfigured");
  assert.equal(reloads, 0, "Connected playback must not reload the page");
  emit(events, "pagehide", {});
}

(async () => {
  const cases = [
    { noToken: true }, { unconfigured: true }, { noFullMode: true }, { cancel: true }, {},
    { cachedToken: true }, { reloadGuards: true },
    { duplicate: true }, { playError: true }, { playError: true, manualRetry: true },
    { emptyQueue: true, expectedPlaybackDiagnostic: { phase: "queue", reason: "queue_mismatch", queue_check: "wrong_length" } },
    { wrongQueue: true, expectedPlaybackDiagnostic: { reason: "queue_mismatch", queue_check: "wrong_track" } },
    { staleQueue: true, expectedPlaybackDiagnostic: { reason: "queue_mismatch", queue_check: "different_queue" } },
    { noQueue: true, expectedPlaybackDiagnostic: { reason: "queue_unavailable", queue_check: "absent" } },
    { expired: true, expectedPlaybackDiagnostic: { phase: "preflight", reason: "expired" } },
    { queueDelay: 90, expectedPlaybackDiagnostic: { phase: "queue", reason: "timeout" } },
    { deferredPlay: true, playEventDelay: 5 }, { rejectPlay: true },
    { wrongPlayingTrack: true, shortDeadline: true, expectedPlaybackDiagnostic: { phase: "confirm", reason: "timeout", track_matches: false } },
    { wrongPlayingTrack: true, shortDeadline: true, earlyTimers: true,
      expectedPlaybackDiagnostic: { phase: "confirm", reason: "timeout", track_matches: false } },
    { wrongPlayingTrack: true, correctTrackLater: true, deferredPlay: true },
    { latePlaying: true, playEventDelay: 90, shortDeadline: true },
    { mediaErrorEvent: true, expectedPlaybackDiagnostic: { reason: "media_error" } },
    { abortDuringPlayback: true, expectedPlaybackDiagnostic: { reason: "cancelled" } }, { emptyPollFirst: true },
    { connectionLoss: 400 }, { connectionLoss: 409 }, { connectionLoss: 503 }, { connectionLoss: "network" },
    { connectionLoss: 400, lossDuringStart: true, playEventDelay: 90, pendingStartDelay: 100 },
    { connectionLoss: "network", lossDuringQueue: true, queueDelay: 90 },
    { connectionLoss: 400, pauseNoEffect: true },
    { connectionLoss: 400, reconnectAfterLoss: true },
    { connectionLoss: 400, connectionLossAfter: 2, rapidReconnect: true, controlSequence: "reconnect_loss", sequencePlaying: true,
      commands: [
        { id: "before-loss-pause", operation: "pause", expires_in_ms: 1500 },
        { id: "before-loss-resume", operation: "resume", expires_in_ms: 20000, afterResult: "before-loss-pause" },
      ] },
    { manualDisconnect: true }, { manualDisconnect: true, pauseNoEffect: true },
    { pauseCommand: true, expectedPlaybackDiagnostic: { phase: "confirm", reason: "confirmed", state: "paused" } }, { pauseCommand: true, pauseIdle: true },
    { pauseCommand: true, pauseIdle: true, pauseThrow: true },
    { pauseCommand: true, pauseIdle: true, pauseReject: true },
    { pauseCommand: true, pauseIdle: true, pauseAuthLostSilently: true,
      expectedPlaybackDiagnostic: { phase: "confirm", reason: "confirmed", state: "paused" } },
    { pauseCommand: true, pauseCompleted: true, pauseAuthLostSilently: true,
      expectedPlaybackDiagnostic: { phase: "confirm", reason: "confirmed", state: "completed" } },
    { pauseCommand: true, pauseAuthLostSilently: true,
      expectedPlaybackDiagnostic: { phase: "confirm", reason: "confirmed", state: "paused" } },
    { pauseCommand: true, pauseNoEffect: true, pauseShort: true, pauseAuthLostSilently: true,
      expectedPlaybackDiagnostic: { phase: "confirm", reason: "authorization_lost", state: "playing" } },
    { pauseCommand: true, pauseStateUnknown: true, pauseNoEffect: true, pauseShort: true, pauseAuthLostSilently: true,
      expectedPlaybackDiagnostic: { phase: "confirm", reason: "authorization_lost", state: "unknown" } },
    { pauseAfterPlay: true, deferredPlay: true, playEventDelay: 90, pauseShort: true, pauseAuthLostSilently: true,
      expectedPlaybackDiagnostic: { phase: "confirm", reason: "authorization_lost" } },
    { pauseCommand: true, pauseThrow: true },
    { pauseCommand: true, pauseLoading: true }, { pauseCommand: true, pauseReject: true },
    { pauseCommand: true, pauseNoEffect: true, pauseShort: true }, { pauseCommand: true, pauseExpired: true },
    { pauseAfterPlay: true, queueDelay: 90 },
    { pauseAfterPlay: true, deferredPlay: true, playEventDelay: 90, pauseShort: true },
    { race: "manual_start", commands: [
      { id: "test-command", operation: "play", track_id: "123", expires_in_ms: 20000 },
      { id: "pause-command", operation: "pause", expires_in_ms: 1500, deliveryDelay: 300 },
    ] },
    { race: "queue_replacement", firstQueueDelay: 90, commands: [
      { id: "test-command", operation: "play", track_id: "123", expires_in_ms: 20000 },
      { id: "pause-command", operation: "pause", expires_in_ms: 1500, deliveryDelay: 5 },
      { id: "second-play", operation: "play", track_id: "456", expires_in_ms: 20000, deliveryDelay: 5 },
    ] },
    { race: "queued_expiry", firstQueueDelay: 90, commands: [
      { id: "test-command", operation: "play", track_id: "123", expires_in_ms: 20000 },
      { id: "second-play", operation: "play", track_id: "456", expires_in_ms: 1550 },
    ] },
    { race: "paused_expiry", playEventDelay: 90, pendingStartDelay: 100, commands: [
      { id: "test-command", operation: "play", track_id: "123", expires_in_ms: 20000 },
      { id: "pause-command", operation: "pause", expires_in_ms: 260, deliveryDelay: 5 },
      { id: "second-play", operation: "play", track_id: "456", expires_in_ms: 1550, deliveryDelay: 5 },
    ] },
    { resumeCommand: true }, { resumeCommand: true, resumeAdvanced: true },
    { resumeCommand: true, resumeEmpty: true, resumeStale: true },
    { resumeCommand: true, resumeAbsent: true, resumeStale: true },
    { resumeCommand: true, resumeStale: true }, { resumeCommand: true, resumeStale: true, resumeInvalidID: true },
    { resumeCommand: true, resumeNoCurrent: true },
    { resumeCommand: true, expired: true }, { resumeCommand: true, rejectPlay: true },
    { resumeCommand: true, playError: true },
    { resumeCommand: true, deferredPlay: true, playEventDelay: 5 },
    { resumeCommand: true, wrongPlayingTrack: true, shortDeadline: true },
    { resumeCommand: true, latePlaying: true, playEventDelay: 90, shortDeadline: true },
    { resumeCommand: true, abortDuringPlayback: true },
    { pauseAfterResume: true, playEventDelay: 90, pendingStartDelay: 100 },
    { race: "resume_paused_expiry", playEventDelay: 90, pendingStartDelay: 100, commands: [
      { id: "test-command", operation: "play", track_id: "123", expires_in_ms: 20000 },
      { id: "pause-command", operation: "pause", expires_in_ms: 260, deliveryDelay: 5 },
      { id: "second-resume", operation: "resume", expires_in_ms: 1550, deliveryDelay: 5 },
    ] },
    { race: "queued_resume_expiry", firstQueueDelay: 90, commands: [
      { id: "test-command", operation: "play", track_id: "123", expires_in_ms: 20000 },
      { id: "second-resume", operation: "resume", expires_in_ms: 1550 },
    ] },
    { controlSequence: "pause_resume_pause", sequencePlaying: true, commands: [
      { id: "first", operation: "pause", expires_in_ms: 1500 },
      { id: "middle", operation: "resume", expires_in_ms: 20000, afterResult: "first" },
      { id: "last", operation: "pause", expires_in_ms: 1500, afterResult: "middle" },
    ] },
    { controlSequence: "resume_pause_resume", commands: [
      { id: "first", operation: "resume", expires_in_ms: 20000 },
      { id: "middle", operation: "pause", expires_in_ms: 1500, afterResult: "first" },
      { id: "last", operation: "resume", expires_in_ms: 20000, afterResult: "middle" },
    ] },
    { controlSequence: "cancel_gated_resume", commands: [
      { id: "first", operation: "resume", expires_in_ms: 20000 },
      { id: "middle", operation: "pause", expires_in_ms: 1500, afterResult: "first" },
      { id: "gated", operation: "resume", expires_in_ms: 20000, afterResult: "middle" },
      { id: "last", operation: "pause", expires_in_ms: 1500, deliveryDelay: 5 },
    ] },
    { controlSequence: "expire_gated_resume", commands: [
      { id: "first", operation: "resume", expires_in_ms: 20000 },
      { id: "middle", operation: "pause", expires_in_ms: 1500, afterResult: "first" },
      { id: "last", operation: "resume", expires_in_ms: 1550, afterResult: "middle" },
    ] },
    { controlSequence: "manual_paused_wake", commands: [
      { id: "first", operation: "resume", expires_in_ms: 20000 },
      { id: "last", operation: "pause", expires_in_ms: 1500, afterResult: "first", manualPause: true },
    ] },
    { controlSequence: "manual_paused_pending_start", deferredPlay: true, commands: [
      { id: "first", operation: "resume", expires_in_ms: 20000 },
      { id: "last", operation: "pause", expires_in_ms: 260, afterResult: "first", manualPause: true },
    ] },
    { controlSequence: "coalesced_pause", sequencePlaying: true, pauseDelay: 80, commands: [
      { id: "first", operation: "pause", expires_in_ms: 1500 },
      { id: "last", operation: "pause", expires_in_ms: 1500, deliveryDelay: 5 },
    ] },
    { queueSDKError: { reason: "CONTENT_UNAVAILABLE", message: "private-sdk-detail", data: { token: "private-token" } },
      expectedPlaybackDiagnostic: { phase: "queue", reason: "queue_rejected", sdk_reason: "CONTENT_UNAVAILABLE" } },
    { queueSDKError: { reason: "private-secret-code", message: "private-sdk-detail" },
      expectedPlaybackDiagnostic: { phase: "queue", reason: "queue_rejected" }, noPlaybackSDKReason: true },
    { rejectPlay: true, playSDKError: { reason: "USER_INTERACTION_REQUIRED", message: "private-sdk-detail" },
      expectedPlaybackDiagnostic: { phase: "confirm", reason: "autoplay_blocked", sdk_reason: "USER_INTERACTION_REQUIRED" } },
    { playError: true, playSDKError: { name: "NotAllowedError", message: "private-sdk-detail" },
      expectedPlaybackDiagnostic: { phase: "play", reason: "autoplay_blocked" }, noPlaybackSDKReason: true },
    { rejectPlay: true, playSDKError: { reason: "TOKEN_EXPIRED", message: "private-sdk-detail" },
      expectedPlaybackDiagnostic: { reason: "authorization_lost", sdk_reason: "TOKEN_EXPIRED" } },
    { rejectPlay: true, playSDKError: throwingProperty("reason"),
      expectedPlaybackDiagnostic: { reason: "play_rejected" }, noPlaybackSDKReason: true },
    { authLostDuringPlayback: true, expectedPlaybackDiagnostic: { reason: "authorization_lost" } },
    { authError: { reason: "UNAUTHORIZED_ERROR", data: { status: 401 }, message: "private-sdk-token" } },
    { authError: { reason: "TOKEN_SECRET_private", status: "401 private-token", message: "private-sdk-token" } },
    { invalidRegion: true }, { sessionError: 503 },
    { reloadAfterError: true, authError: { reason: "UNAUTHORIZED_ERROR", data: { status: 401 }, message: "private-sdk-token" } },
    { diagnostics: true, authEvents: [{ message: appleMessage("unavailable") }, { status: 0 }],
      authError: { reason: "AUTHORIZATION_ERROR", message: "private-sdk-token" },
      expectedDiagnostics: ["authorization_started", "apple_unavailable", "status_0", "authorization_rejected"] },
    { diagnostics: true, authEvents: [{ message: appleMessage("authorize") }],
      authError: { reason: "AUTHORIZATION_ERROR", message: "private-sdk-token" },
      expectedDiagnostics: ["authorization_started", "apple_authorize", "authorization_rejected"] },
    { diagnostics: true, authError: { reason: "AUTHORIZATION_ERROR", message: "private-sdk-token" },
      expectedDiagnostics: ["authorization_started", "authorization_rejected"] },
    { diagnostics: true, authEvents: [{ status: 0 }],
      authError: { reason: "AUTHORIZATION_ERROR", message: "private-sdk-token" },
      expectedDiagnostics: ["authorization_started", "status_0", "authorization_rejected"] },
    { diagnostics: true, authEvents: [{ message: appleMessage("authorize") }, { status: 3 }],
      expectedDiagnostics: ["authorization_started", "apple_authorize", "status_3", "authorization_resolved", "local_registration_started", "local_registration_succeeded"] },
    { authEvents: [{ message: appleMessage("unavailable") }, { status: 0 }],
      authError: { reason: "AUTHORIZATION_ERROR", message: "private-sdk-token" }, expectedDiagnostics: [] },
    { diagnosticsSearch: "?diagnostics=0", authEvents: [{ message: appleMessage("unavailable") }],
      authError: { reason: "AUTHORIZATION_ERROR", message: "private-sdk-token" }, expectedDiagnostics: [] },
    { diagnostics: true, authEvents: [
        { message: appleMessage("unavailable", { origin: "https://authorize.music.apple.com.evil.example" }) },
        { message: appleMessage("unavailable", { origin: "http://authorize.music.apple.com" }) },
        { message: appleMessage("unavailable", { data: JSON.stringify({ jsonrpc: "2.0", method: "unavailable" }) }) },
        { message: appleMessage("unavailable", { data: [{ jsonrpc: "2.0", method: "unavailable" }] }) },
        { message: appleMessage("unavailable", { data: null }) },
        { message: appleMessage("unavailable", { data: { jsonrpc: "1.0", method: "unavailable" } }) },
        { message: appleMessage("unavailable", { data: { jsonrpc: "2.0", method: "private-diagnostic-token" } }) },
        ...["0", -2, 4, true, null, NaN].map((status) => ({ status })),
      ], authError: { reason: "AUTHORIZATION_ERROR", message: "private-sdk-token" },
      expectedDiagnostics: ["authorization_started", "authorization_rejected"] },
    { diagnostics: true, authEvents: [
        ...["thirdPartyInfo", "decline", "switchUserId", "close"].map((method) => ({ message: appleMessage(method) })),
        ...[-1, 0, 1, 2, 3].map((status) => ({ status })),
      ], authError: { reason: "AUTHORIZATION_ERROR", message: "private-sdk-token" },
      expectedDiagnostics: ["authorization_started", "apple_third_party_info", "apple_decline", "apple_switch_user", "apple_close", "status_-1", "status_0", "status_1", "status_2", "status_3", "authorization_rejected"] },
    { diagnostics: true, authEvents: [
        { message: throwingProperty("origin") },
        { message: throwingProperty("data", { origin: appleOrigin }) },
        { message: appleMessage("unavailable", { data: throwingProperty("jsonrpc") }) },
        { message: appleMessage("unavailable", { data: throwingProperty("method", { jsonrpc: "2.0" }) }) },
        { statusEvent: throwingProperty("authorizationStatus") },
        { message: appleMessage("unavailable", { data: throwingProperty("params", { jsonrpc: "2.0", method: "unavailable" }) }) },
      ], authError: { reason: "AUTHORIZATION_ERROR", message: "private-sdk-token" },
      expectedDiagnostics: ["authorization_started", "apple_unavailable", "authorization_rejected"] },
    { diagnostics: true, diagnosticFlood: true,
      authEvents: Array.from({ length: 100 }, () => ({ message: appleMessage("unavailable") })),
      authError: { reason: "AUTHORIZATION_ERROR", message: "private-sdk-token" },
      expectedDiagnostics: ["authorization_started", ...Array(37).fill("apple_unavailable")] },
  ];
  for (const options of cases) await scenario(options);
  console.log(`${cases.length} MusicKit browser mock scenarios passed`);
})().catch((error) => { console.error(error); process.exitCode = 1; });
