# Connect Apple Music for a local test

This integration searches the full Apple Music catalog and plays through Apple's MusicKit JS player in a browser on this Mac. It does not use Music.app scripting or require songs to be in your library. Keep the local music server and its player tab open while using APM. A compatible browser, Apple Music account authorization, and an eligible subscription are required for full-track playback.

## Apple developer setup

Use your Apple Developer Program account to create a **Media ID** with **MusicKit** enabled, then a **Media Services key** associated with that Media ID. Download its `.p8` private key and keep it on this Mac. Apple only offers that key download once. Record your account's **Team ID** and the new **Key ID**. See Apple's [Media ID and key instructions](https://developer.apple.com/help/account/capabilities/create-a-media-identifier-and-private-key).

Create `work/apple-music.json` with the three fields below, replacing every placeholder. The private-key path may be absolute or relative to this JSON file. Do not paste the private-key contents into chat, put them in browser JavaScript, or commit them to Git.

```json
{
  "team_id": "YOURTEAMID",
  "key_id": "YOURKEYID0",
  "private_key_path": "/absolute/path/to/AuthKey_YOURKEYID0.p8"
}
```

The project ignores `work/` and `.p8` files. The signing key stays on the server. The server issues one-hour ES256 developer tokens; Apple's browser SDK manages user authorization. APM does not send Apple Music user tokens to Gemma or store them in the assistant database.

## Start the player and assistant

```sh
uv pip install --python .venv/bin/python -e '.[apple-music]'
.venv/bin/python -m apm.apple_music_server
```

Open the **Apple Music connection** URL printed in that terminal. The fragment contains a local connection token, which the page moves into session storage before loading Apple's SDK. Click **Connect Apple Music** and finish Apple's authorization. Playback may require an explicit browser click before voice-initiated playback is allowed. If prompted, use the page's playback control; a blocked or uncertain request is never automatically retried.

Use the page's song and artist fields to test a recording, then start APM in another terminal:

```sh
.venv/bin/apm --voice
```

The server saves its loopback address and a random local API token in `work/music-connection.json` with owner-only file permissions. APM and `apm-server` read that file at startup. A normal server launch generates a new token, requiring those clients to restart and the browser to open the newly printed connection link. To restore a stopped server while keeping existing clients' credentials, run this in a separate terminal and leave it running:

```sh
.venv/bin/apm-music --reuse-connection
```

Then reconnect the existing browser player. This requires a valid saved connection matching the selected port; it does not silently generate a replacement token. The server's queue and browser session still reset on restart. Removing the connection file restores the unconfigured default. The server only binds to `127.0.0.1`; do not expose this test server on the LAN.

Say “Hey Gemma,” wait for the listening cue, then request a song and artist. Ambiguous recordings are presented as choices. The player reports **playing** only after observing the selected Apple catalog ID in the active player state. Acknowledgement without that observation remains unconfirmed. Browser closure, network errors, expired sessions, and missing authorization produce failures rather than simulated success.

During playback, a detected “Hey Gemma” pauses the browser player before APM captures the command. Wait for the listening cue after the pause. Music remains paused until a new play request. When Gemma asks a question, answer after it finishes speaking without repeating the wake phrase; the default reply window is eight seconds. After upgrading this feature, restart the music server and voice client, then reload and reconnect the player page so both ends understand pause commands. This cleans up command capture but does not identify speakers or cancel music at the wake detector.

The current player advertises protocol version 3 when connecting, supporting both pause and resume. Version 2 supports pause only; version 1 supports neither control. An unsupported control returns `unknown` with `player_update_required`, rather than claiming success. Voice mode stops capture and returns to text after a single unconfirmed wake-time pause, so a broken connection cannot repeatedly trigger the same warning. Reload and reconnect the player, then enter `/voice` again. An already-quiet player confirms its idle state without invoking Apple's pause method unnecessarily.

Losing the local connection requests a pause and cancels pending starts, including when another player tab replaces the session. A failed command poll also stops automatic recovery: reconnect explicitly before playing again. APM does not treat a disconnected player as proof of silence; voice capture stays closed until the selected player reconnects and confirms pause. Pause or close any old player tabs as well. Voice still works normally when no music provider is configured.

Pause controls local media and does not require a valid Apple Music authorization. If sign-in ages out while playback is completed or paused, the player can still confirm that it is quiet and voice questions continue normally. Active playback must actually stop before confirmation, and an unresolved startup still blocks confirmation. New play/resume requests continue to require authorization. Even when authorization is lost, pause failures are reported to the local bridge instead of silently dropping the response.

MusicKit v3 suppresses repeated public `play()` or `pause()` calls within 250 ms. The player spaces actual calls to the same method by at least 255 ms, coalescing pending pauses and preserving each command's original deadline. This only delays closely spaced repeats; it does not add a fixed delay to ordinary controls or retry playback. A newer pause cancels a resume still waiting to invoke the SDK.

A pause also cancels earlier song requests that are still searching the catalog. Their eventual search results cannot restart playback; a new explicit play or resume request is required.

To test the controls, play a song, say “Hey Gemma, pause the music,” and wait for “Music paused.” Then say “Hey Gemma, continue playing the music.” Resume uses the currently loaded song and position without repeating the catalog search. If the page was refreshed and its queue is empty, request a song first. After installing these tools, restart APM once so Gemma receives the new tool definitions, and refresh/reconnect the player page to load resume support.

If the player reports `player_update_required`, refresh the existing browser tab with its page-reload control (Cmd-R on Mac), then click **Connect Apple Music**. Restarting Python does not reload JavaScript in an open tab. Older pages have a **Reload setup** button that only refreshes credentials; use the browser's page reload to upgrade those pages. Updated pages label this control **Reload player** and reload the whole page.

## Troubleshoot authorization

The player document sends `Referrer-Policy: strict-origin`. Apple's authorization page uses `document.referrer` to establish the callback destination, even when the SDK also supplies a return URL. A `no-referrer` policy on the player can let sign-in and consent finish but prevent the popup from returning authorization. The origin-only policy shares the local scheme, hostname, and port, excluding the path, query, and fragment. API and script responses retain `no-referrer`.

For a failed sign-in, add `?diagnostics=1` before the connection link's `#token=…` fragment, or to the existing player URL in the same tab. The optional **Connection diagnostics** panel observes the next Connect attempt. It keeps only bounded event names, status codes, and relative times on the page; it never sends this trace to the backend or records callback token payloads. A reload clears the trace. Apple callback observation uses an internal MusicKit v3 protocol and is diagnostic evidence only: a callback's arrival does not establish token validity, and status `0` can be cleanup after a different failure. An Apple `unavailable` response still needs Apple's underlying response to explain the cause.

For playback failures, expand **Latest playback diagnostics**. This records the command's stage, fixed failure reason, elapsed milliseconds, player state, and whether the requested track and queue matched. Accepted completions also appear under `player.last_command` in the authenticated local status endpoint. These diagnostics exclude song metadata, identifiers, account details, tokens, URLs, and raw SDK messages. They distinguish queue loading, browser autoplay restrictions, media errors, and expired commands; they do not change playback confirmation or retry uncertain commands. Session replacement clears the server's last diagnostic.

If a command expires without an accepted browser result, the bridge records whether it was awaiting `delivery` or `completion`, along with elapsed time. A late result can be noted as having arrived, but cannot revive the expired command or overwrite a newer completion. This distinguishes a missing browser response from an SDK failure without exposing command identifiers.

## Limits of this first integration

- Audio plays in the connected browser. Independent smart-speaker routing is not implemented.
- Keep one connected player page open. Connecting another page replaces the previous session and cancels pending requests.
- Browser autoplay and DRM support vary. If Apple's player cannot start audio, use a supported browser and its explicit playback control.
- The developer token lasts one hour. Reload and reconnect the player to obtain a fresh token for longer sessions.
- Authorization does not prove subscription eligibility or actual audio output. Full playback still depends on Apple's player and the account's capabilities.
- This prototype does not download music or implement a mobile login flow.

The provider is in `apm/musickit.py`, developer-token signing in `apm/apple_music_tokens.py`, the local server in `apm/apple_music_server.py`, and the assistant client in `apm/music_connection.py`. Apple's [MusicKit documentation](https://developer.apple.com/musickit/) describes the supported player and authorization flow.
