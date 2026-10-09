"""Provider-neutral music requests; catalog IDs remain inside MusicService."""
from copy import deepcopy


def _tool(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": deepcopy(properties),
                       "required": list(required), "additionalProperties": False}}}


_TEXT = {"type": "string", "minLength": 1, "maxLength": 200, "pattern": r"\S"}
_QUERY = {
    "title": _TEXT,
    "artist": _TEXT,
    "version": {"type": "string", "enum": [
        "studio", "live", "remix", "acoustic", "karaoke", "instrumental", "remaster"]},
}

MUSIC_TOOLS = [
    _tool("pause_music",
          "Pause or stop the current music playback. Use for 'pause the music', 'stop the song', "
          "or 'stop playing'. Takes no task, timer, song, or selection ID. "
          "The wake phrase may already have paused playback; this operation is safe to repeat.",
          {}, []),
    _tool("resume_music",
          "Resume or continue the music already loaded in the player, from its current position. "
          "Use for 'continue playing the music', 'resume the song', or 'play again' referring to paused playback. "
          "Takes no IDs or song title; does not search for a new song or restart it from the beginning. "
          "Report success only when playback is confirmed.",
          {}, []),
    _tool("resolve_music",
          "Search the connected music catalog for a song title and optional artist/version. "
          "Does not play music. Use returned matches or clarification choices; never invent tracks or provider IDs.",
          _QUERY, ["title"]),
    _tool("play_music",
          "For an explicit user request to play a song, search the connected catalog by title "
          "and optional artist/version, then automatically play the first best matching recording. "
          "For 'play <song> by <artist>', put the song in title and the artist in artist, even for unfamiliar names. "
          "Do not ask the user to choose among albums or search again after success. "
          "Do not claim playback unless the result confirms it.",
          _QUERY, ["title"]),
    _tool("play_music_selection",
          "Play the user's chosen clarification result using its exact opaque selection_id "
          "returned by the music resolver. Use only after the user chooses that result; "
          "never invent selection IDs or pass provider track IDs.",
          {"selection_id": {"type": "string", "minLength": 1, "maxLength": 128, "pattern": r"\S"}},
          ["selection_id"]),
]
