"""Token signing with temporary generated keys, never real Apple credentials."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
import jwt

from apm.apple_music_tokens import DeveloperToken


class DeveloperTokenTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.settings_path = self.directory / "settings.json"
        self.key_path = self.directory / "test-private.p8"
        self.key = ec.generate_private_key(ec.SECP256R1())
        self.write_key(self.key)
        self.settings = {"team_id": "TEAMID1234", "key_id": "KEYID12345", "private_key_path": self.key_path.name}
        self.write_settings()
        self.now = 2_000_000_000
        self.tokens = DeveloperToken(self.settings_path, now=lambda: self.now)

    def write_key(self, key, password=None):
        protection = serialization.NoEncryption() if password is None else serialization.BestAvailableEncryption(password)
        self.key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                                   serialization.PrivateFormat.PKCS8, protection))

    def write_settings(self):
        self.settings_path.write_text(json.dumps(self.settings))

    def decode(self, token, key=None):
        return jwt.decode(token, (key or self.key).public_key(), algorithms=["ES256"],
                          options={"verify_exp": False, "verify_iat": False})

    def test_relative_key_signs_valid_es256_claims_and_status_has_no_credentials(self):
        token = self.tokens()
        self.assertEqual(self.decode(token), {"iss": "TEAMID1234", "iat": self.now, "exp": self.now + 3600})
        header = jwt.get_unverified_header(token)
        self.assertEqual((header["alg"], header["kid"]), ("ES256", "KEYID12345"))
        status = self.tokens.status()
        self.assertTrue(status["configured"])
        self.assertEqual(set(status), {"configured", "message"})
        for private in (token, str(self.key_path), "PRIVATE KEY", "TEAMID1234", "KEYID12345"):
            self.assertNotIn(private, json.dumps(status) + repr(self.tokens))

    def test_constructor_is_lazy_and_missing_setup_is_actionable(self):
        self.settings_path.unlink()
        tokens = DeveloperToken(self.settings_path)
        self.assertFalse(tokens.status()["configured"])
        self.assertIn("team_id", tokens.status()["message"])
        with self.assertRaisesRegex(ValueError, "settings file"):
            tokens()

    def test_cache_renews_at_sixty_seconds_before_expiry(self):
        with patch("apm.apple_music_tokens.jwt.encode", wraps=jwt.encode) as encode:
            first = self.tokens()
            self.now += 3539
            self.assertEqual(self.tokens(), first)
            self.now += 1
            renewed = self.tokens()
            self.assertNotEqual(renewed, first)
            self.assertEqual(encode.call_count, 2)
            self.assertEqual(self.decode(renewed)["iat"], self.now)

    def test_credential_rotation_invalidates_cached_token_and_removal_blocks_it(self):
        original = self.tokens()
        rotated_key = ec.generate_private_key(ec.SECP256R1())
        self.write_key(rotated_key)
        rotated = self.tokens()
        self.assertNotEqual(rotated, original)
        self.assertEqual(self.decode(rotated, rotated_key)["iss"], self.settings["team_id"])
        self.settings["key_id"] = "NEWKEY1234"
        self.write_settings()
        self.assertEqual(jwt.get_unverified_header(self.tokens())["kid"], "NEWKEY1234")
        self.key_path.unlink()
        self.assertFalse(self.tokens.status()["configured"])
        with self.assertRaisesRegex(ValueError, "could not be read"):
            self.tokens()

    def test_malformed_configuration_errors_do_not_echo_private_values(self):
        malformed = [None, [], {}, {**self.settings, "unexpected": "private-marker"},
                     {**self.settings, "team_id": "private-marker"},
                     {**self.settings, "key_id": "keyid12345"},
                     {**self.settings, "private_key_path": False},
                     {**self.settings, "private_key_path": "missing-private-marker.p8"},
                     {**self.settings, "private_key_path": "~apm_no_such_test_user/private-marker.p8"}]
        for config in malformed:
            with self.subTest(config=config):
                self.settings_path.write_text(json.dumps(config))
                status = self.tokens.status()
                self.assertFalse(status["configured"])
                self.assertNotIn("private-marker", json.dumps(status))
                self.assertNotIn(str(self.directory), json.dumps(status))
                with self.assertRaises(ValueError) as raised:
                    self.tokens()
                self.assertNotIn("private-marker", str(raised.exception))
        self.settings_path.write_text('{"private-marker":')
        self.assertFalse(self.tokens.status()["configured"])

    def test_status_validates_actual_key_type_and_encryption(self):
        self.key_path.write_text("private-marker malformed private key")
        self.assertFalse(self.tokens.status()["configured"])
        self.assertNotIn("private-marker", self.tokens.status()["message"])
        self.write_key(ec.generate_private_key(ec.SECP384R1()))
        self.assertFalse(self.tokens.status()["configured"])
        self.write_key(self.key, password=b"test-only-password")
        self.assertFalse(self.tokens.status()["configured"])
        self.write_key(self.key)
        with patch("apm.apple_music_tokens.jwt.encode", side_effect=AssertionError("must not sign during status")):
            self.assertTrue(self.tokens.status()["configured"])

    def test_concurrent_requests_share_one_signed_token(self):
        with patch("apm.apple_music_tokens.jwt.encode", wraps=jwt.encode) as encode:
            with ThreadPoolExecutor(max_workers=4) as executor:
                tokens = list(executor.map(lambda _: self.tokens(), range(8)))
            self.assertEqual(len(set(tokens)), 1)
            self.assertEqual(encode.call_count, 1)

    def test_invalid_clock_and_signing_errors_are_sanitized(self):
        for instant in (True, float("nan"), float("inf"), -1, "private-marker"):
            with self.subTest(instant=instant), self.assertRaisesRegex(ValueError, "Unix time"):
                DeveloperToken(self.settings_path, now=lambda: instant)()
        with patch("apm.apple_music_tokens.jwt.encode", side_effect=RuntimeError("private-marker")):
            with self.assertRaises(ValueError) as raised:
                self.tokens()
        self.assertNotIn("private-marker", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
