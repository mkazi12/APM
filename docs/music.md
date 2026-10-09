# Service-neutral music requests

For “play i'm on fire by bruce springstein,” the intended catalog match is **I’m On Fire — Bruce Springsteen**, as listed on the [artist’s official track page](https://brucespringsteen.net/track/im-on-fire/). The implementation uses that metadata in test fixtures with synthetic IDs; no artist-specific correction is hardcoded.

Gemma extracts title, artist, and any requested version. `play_music` searches the catalog and plays the highest scoring eligible match. Ties use the provider's result order, so multiple compilation releases of the same song do not require an album choice. `resolve_music` searches without playback and preserves choices for browsing. When a choice is needed, the backend returns short-lived selection IDs; after the user chooses, `play_music_selection` forwards the corresponding verified catalog ID to the provider.

## Current connection state

The default `MusicService()` has no provider. Requests return `not_configured`; they do not claim that a song played. A full-catalog Apple Music adapter and local MusicKit player are available through the [Apple Music setup guide](apple-music.md). They require developer configuration and account authorization. Starting that music server saves a connection used by APM on its next startup. Other service adapters are not implemented.

The live Gemma regression test uses a fake catalog/player. It demonstrates natural-language parsing, typo handling, catalog selection, and dispatch to the correct synthetic ID. It does not establish real account authorization, availability in a user's country, speaker routing, or successful audio output.

## App API

Run the existing `apm-server`; the music endpoints use its existing local-only access and optional bearer authentication.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/v1/music/status` | Whether a provider is connected |
| POST | `/v1/music/resolve` | Find a recording without playback |
| POST | `/v1/music/play` | Search and play the top eligible match |
| POST | `/v1/music/pause` | Pause the connected player and confirm it is quiet; empty JSON body |
| POST | `/v1/music/resume` | Resume the current queue from its position; empty JSON body |
| POST | `/v1/music/selections/{selection_id}/play` | Play a previously returned choice |

Search/play body:

```json
{"title":"I'm on fire","artist":"Bruce Springstein"}
```

`artist` and `version` are optional; title and artist accept up to 200 printable characters. Supported version requests are `studio`, `live`, `remix`, `acoustic`, `karaoke`, `instrumental`, and `remaster`. Use selection playback only with an ID returned by this backend. Sending a provider URL or guessed track ID is not supported.

Resolution statuses are `matched`, `ambiguous`, `not_found`, `not_configured`, and `failed`. Playback can return `playing`, `accepted`, `failed`, or `unknown`, along with unresolved search outcomes. `playing` requires the provider to report both active playback and the selected track ID. Acknowledgement alone produces `accepted`. Timeouts or mismatched observed tracks remain unconfirmed, and writes are never automatically retried.

Selection tokens last ten minutes, are stored only in the current service process, and are consumed before attempting playback. A restart, expiry, or uncertain attempt requires a fresh search. The Apple Music connection routes voice and app requests to one local music server, allowing both clients to use the same selections. Without a saved connection, each process has an unconfigured service.

## Matching rules

The resolver normalizes Unicode, case, apostrophes, accents, and punctuation, then compares title and requested artist separately. Similarity scores are heuristics, not calibrated confidence probabilities. Both must meet the matching thresholds. Playback chooses the highest scoring eligible candidate, preserving the provider's order when scores tie. When no artist is supplied, this can choose among different artists' recordings of the same title. Supply an artist to constrain the match, or use `resolve_music` to browse alternatives. Search-only resolution still returns choices when close competitors lack a clear margin.

If Gemma puts an entire request such as `Thread by Ear` into the title field without an artist, the resolver searches that literal title first. Only when it finds no eligible match does it split at the last ` by ` and try one title/artist search. A matching literal title such as `Stand by Me` stays intact, and an explicitly supplied artist disables this fallback.

Playback prefers a studio recording when no version is specified. A live-only result, unknown version, or mixed version label still needs a choice. Explicit version requests and recognized title suffixes are honored. Preserve specific details in the title, such as `Song (2010 Remaster)` or `Song (Live at Rome 2013)`; the version field alone cannot express them. Conflicting catalog version labels cannot produce an automatic match. Unplayable tracks are excluded. Duplicate catalog IDs are collapsed, and different editions are merged only when the provider supplies a consistent recording identity; independent compilation editions can still be selected automatically for playback.

Providers return at most 50 candidates; clarification responses show at most ten choices. Matching only works on returned candidates. An adapter must implement useful search, including typo-tolerant retrieval or a broader fallback when a strict title/artist query returns nothing. The generic resolver cannot recover a track omitted by its provider.

## Add a provider later

Implement the `MusicProvider` protocol in a separate adapter:

```python
class ExampleProvider:
    name = "Example"

    def search(self, title, artist=None):
        # Return list[Track] from the provider's catalog.
        ...

    def play(self, track_id):
        # Perform one explicit write, then report acknowledgement and readback.
        # Return PlaybackResult(accepted=..., playing=..., track_id=...).
        ...

    def pause(self):
        # Confirm playing=False, and report whether playback was active.
        # Return PlaybackResult(accepted=True, playing=False, was_playing=...).
        ...

    def resume(self):
        # Resume the existing queue without replacing it or seeking.
        # Confirm playing=True and the current provider track_id, or report
        # PlaybackResult(False, False, None) when there is nothing loaded.
        ...

    def close(self):
        ...
```

`Track` includes the provider ID, title, artist tuple, album, version, playable flag, and optional stable recording identity. `PlaybackResult` keeps acknowledgement separate from observation. Inject `MusicService(provider)` into `AssistantController(home, tasks, music=music)` or `create_app(home, tasks=tasks, music=music)`. The owner closes the injected service; the server lifecycle already does so.

Voice mode calls `MusicService.pause()` directly after wake detection, before sending speech to Gemma. A confirmed `paused` response requires `playing: false`; `was_playing` is true, false, or unknown. An absent or disconnected provider returns `unavailable`; an attempted but unconfirmed pause returns `unknown`. Adapters without `pause()` remain usable, but cannot pause on wake. Apple Music pause preempts pending playback and has a short deadline; no automatic retry or resume is performed.

Gemma also has explicit `pause_music` and `resume_music` tools, both with empty arguments. Music controls never use scheduled-task UUIDs or catalog selections. `resume_music` returns `resumed` only after the provider confirms playback of the loaded queue item, `empty` when nothing is loaded, or an unavailable/unknown result without claiming success. It performs no catalog search, queue replacement, or seek. Explicit pause remains valid after wake detection has already paused music. A confirmed control ends that conversation turn without opening an unsolicited follow-up question.

The adapter owns authentication, token storage, account/market availability, target-device selection, timeouts, and safe readback. The generic module never opens URLs, passes arbitrary model-generated IDs to a player, or logs raw provider exception details. Future adapters should follow their service's official catalog and playback APIs, such as [Spotify search](https://developer.spotify.com/documentation/web-api/reference/search) or [Apple Music catalog search](https://developer.apple.com/documentation/applemusicapi/search-for-catalog-resources-(by-type)).

Tool schemas are kept in `apm/toolsets/music.py`; matching/provider contracts are in `apm/music.py`; the shared execution boundary remains in `apm/assistant.py`. Tests cover service logic, API calls, and model context without actual music playback.
