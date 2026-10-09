"""Local Apple Music developer-token signing; no Apple account requests.

Apple's ES256 developer-token contract:
https://developer.apple.com/documentation/applemusicapi/generating-developer-tokens
The private key stays on the server. Only the signed, short-lived developer JWT
is intended for MusicKit JS. A Music User Token is managed separately by MusicKit.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from threading import Lock
import time

from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
import jwt


_IDENTIFIER = re.compile(r"[A-Z0-9]{10}")
_LIFETIME = 3600
_RENEW_BEFORE = 60


class DeveloperToken:
    """Lazy credentials loaded from JSON; cached JWTs renew one minute early.

    Credentials are validated on each use, so a removed or invalid key cannot
    silently leave an old cached token available. Valid credential changes also
    invalidate the cache. Neither keys nor tokens appear in status or repr.
    """

    def __init__(self, settings_path, *, now=None):
        self.settings_path = Path(settings_path).expanduser()
        self._now = time.time if now is None else now
        self._lock = Lock()
        self._cached = None

    def _load(self):
        try:
            settings = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise ValueError("Create the Apple Music settings file with team_id, key_id, and private_key_path.") from None
        except (OSError, ValueError, UnicodeError):
            raise ValueError("Apple Music settings must be a readable JSON file.") from None
        if not isinstance(settings, dict) or set(settings) != {"team_id", "key_id", "private_key_path"}:
            raise ValueError("Apple Music settings require team_id, key_id, and private_key_path only.")
        for field in ("team_id", "key_id"):
            if not isinstance(settings[field], str) or not _IDENTIFIER.fullmatch(settings[field]):
                raise ValueError("Apple Music team_id and key_id must each contain 10 uppercase letters or digits.")
        supplied_path = settings["private_key_path"]
        if not isinstance(supplied_path, str) or not supplied_path.strip() or "\x00" in supplied_path:
            raise ValueError("Apple Music private_key_path must name a local .p8 private key file.")
        try:
            key_path = Path(supplied_path).expanduser()
            if not key_path.is_absolute():
                key_path = self.settings_path.parent / key_path
            pem = key_path.read_bytes()
        except (OSError, RuntimeError, ValueError):
            raise ValueError("Apple Music private key could not be read; check private_key_path and file access.") from None
        try:
            key = serialization.load_pem_private_key(pem, password=None)
        except (ValueError, TypeError, UnsupportedAlgorithm):
            raise ValueError("Apple Music requires an unencrypted P-256 EC private key in PEM (.p8) format.") from None
        if not isinstance(key, ec.EllipticCurvePrivateKey) or not isinstance(key.curve, ec.SECP256R1):
            raise ValueError("Apple Music requires a P-256 EC private key.")
        public_key = key.public_key().public_bytes(serialization.Encoding.DER,
                                                 serialization.PublicFormat.SubjectPublicKeyInfo)
        fingerprint = (settings["team_id"], settings["key_id"], hashlib.sha256(public_key).digest())
        return settings["team_id"], settings["key_id"], key, fingerprint

    def status(self):
        """Validate setup without issuing or disclosing a developer token."""
        try:
            self._load()
        except ValueError as exc:
            return {"configured": False, "message": str(exc)}
        return {"configured": True, "message": "Apple Music developer credentials are ready."}

    def __call__(self):
        with self._lock:
            team_id, key_id, key, fingerprint = self._load()
            try:
                instant = self._now()
                if (isinstance(instant, bool) or not isinstance(instant, (int, float))
                        or not math.isfinite(instant) or instant < 0):
                    raise ValueError
                issued_at = int(instant)
            except (ValueError, TypeError, OverflowError):
                raise ValueError("Apple Music token signing requires a valid current Unix time.") from None
            if self._cached is not None:
                old_fingerprint, old_issued, expires, token = self._cached
                if fingerprint == old_fingerprint and old_issued <= issued_at < expires - _RENEW_BEFORE:
                    return token
            expires = issued_at + _LIFETIME
            try:
                token = jwt.encode({"iss": team_id, "iat": issued_at, "exp": expires}, key,
                                   algorithm="ES256", headers={"kid": key_id})
            except Exception:
                raise ValueError("Apple Music developer token could not be signed; check the configured credentials.") from None
            self._cached = (fingerprint, issued_at, expires, token)
            return token
