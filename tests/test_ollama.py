import base64
import io
import json
from copy import deepcopy
import unittest
from unittest.mock import patch
from urllib.error import URLError

from apm.ollama_backend import OllamaBackend, OllamaError, Transport, encode_audio
from apm.tools import SimulatedHome

CALL = {"function": {"name": "set_lights", "arguments": {"device": "kitchen_lights", "on": False}}}

class FakeTransport:
    def __init__(self, chunks=None, capabilities=None):
        self.chunks = chunks or []
        self.sent = []
        self.capabilities = capabilities if capabilities is not None else ["tools", "audio"]
    def request(self, path, payload):
        self.sent.append((path, payload))
        return {"capabilities": self.capabilities} if path == "/api/show" else {"done": True}
    def stream(self, path, payload):
        self.sent.append((path, payload))
        yield from self.chunks

class OllamaTests(unittest.TestCase):
    def make(self, chunks):
        t = FakeTransport(chunks)
        b = OllamaBackend(transport=t)
        b.prepare()
        return b,t

    def test_streaming_and_tool_results_are_preserved_in_history(self):
        b,t = self.make([
            {"message": {"thinking": "hidden", "content": "Working "}},
            {"message": {"content": "on it.", "tool_calls": [CALL]}},
            {"done": True, "done_reason": "stop", "eval_count": 12}])
        pieces=[]
        response,timing=b.predict(text="turn off kitchen lights",on_text=pieces.append)
        self.assertEqual(''.join(pieces), 'Working on it.')
        results=SimulatedHome().execute(response['tool_calls'])
        b.commit(response,results)
        b.predict(text="what about now?")
        messages=t.sent[-1][1]['messages']
        self.assertEqual(messages[-2]['role'],'tool')
        self.assertIn('"state": "off"',messages[-2]['content'])
        self.assertFalse(t.sent[-1][1]['think'])
        self.assertEqual(t.sent[-1][1]['keep_alive'],'10m')
        self.assertEqual(timing['generated_tokens'],12)

    def test_incomplete_or_truncated_stream_is_not_executable(self):
        for ending in [[], [{"done": True, "done_reason": "length"}]]:
            with self.subTest(ending=ending):
                b,_=self.make([{"message":{"tool_calls":[CALL]}}]+ending)
                with self.assertRaises(OllamaError): b.predict(text="off")
                self.assertIsNone(b.pending)
                self.assertEqual(b.turns,[])

    def test_history_trims_whole_turns_and_reset_clears(self):
        b,_=self.make([{"message":{"tool_calls":[CALL]},"done":True}])
        for i in range(6):
            response,_=b.predict(text=str(i))
            b.commit(response,SimulatedHome().execute(response['tool_calls']))
        self.assertEqual(len(b.turns),4)
        self.assertEqual(b.turns[0][0]['content'],'2')
        self.assertTrue(all(turn[-1]['role']=='tool' for turn in b.turns))
        b.reset()
        self.assertEqual(b.turns,[])

    def test_audio_is_native_wav_attachment(self):
        import numpy as np
        import soundfile as sf
        b,t=self.make([{"message":{"content":"hi"},"done":True}])
        b.predict(audio=np.zeros(1600,dtype=np.float32))
        message=t.sent[-1][1]['messages'][-1]
        data=base64.b64decode(message['images'][0])
        self.assertEqual(data[:4],b'RIFF')
        samples,rate=sf.read(io.BytesIO(data))
        self.assertEqual(rate,16000)
        self.assertEqual(len(samples),1600)
        self.assertEqual(len(t.sent[-1][1]['tools']),3)

    def test_fresh_context_follows_history_without_rewriting_the_cached_prefix(self):
        from apm.assistant import TASK_TOOLS
        from apm.home import HomeController
        from apm.toolsets.music import MUSIC_TOOLS
        b,t = self.make([{"message":{"tool_calls":[CALL]},"done":True}])
        context = HomeController().context()
        context["tools"] += deepcopy(TASK_TOOLS + MUSIC_TOOLS)
        context.update(clock={"utc":"2026-10-07T12:00:00+00:00","timezone":"UTC"},
                       scheduled_tasks=[], music={"configured":True,"provider":"fixture"})
        b.configure_home(context)
        response,_ = b.predict(text="turn off kitchen lights")
        first = deepcopy(t.sent[-1][1])
        b.commit(response, SimulatedHome().execute(response["tool_calls"]))
        context["clock"]["utc"] = "2026-10-07T12:01:00+00:00"
        context["scheduled_tasks"] = [{"id":"actual-task","name":"Tea","status":"scheduled"}]
        context["music"]["configured"] = False
        b.configure_home(context)
        b.predict(text="What about now?")
        second = t.sent[-1][1]
        self.assertEqual(first["messages"][0], second["messages"][0])
        self.assertEqual(first["tools"], second["tools"])
        self.assertEqual([message["role"] for message in second["messages"]],
                         ["system", "user", "assistant", "tool", "system", "user"])
        fresh = second["messages"][-2]["content"]
        self.assertIn("12:01:00", fresh)
        self.assertIn('"id": "actual-task"', fresh)
        self.assertIn('"configured": false', fresh)
        self.assertNotIn("12:00:00", json.dumps(second["messages"]))
        self.assertNotIn("Current clock", json.dumps(b.turns))
        self.assertIn('"state": "off"', second["messages"][-3]["content"])

    def test_completed_voice_turn_keeps_original_audio_and_actual_results(self):
        import numpy as np
        b,t = self.make([{"message":{"tool_calls":[CALL]},"done":True}])
        response,_ = b.predict(audio=np.zeros(1600,dtype=np.float32))
        original = deepcopy(t.sent[-1][1]["messages"][-1])
        b.commit(response, SimulatedHome().execute(response["tool_calls"]))
        b.predict(text="What exactly did I ask for?")
        messages = t.sent[-1][1]["messages"]
        self.assertEqual(messages[1], original)
        self.assertEqual(messages[2]["tool_calls"], [CALL])
        self.assertIn('"state": "off"', messages[3]["content"])

    def test_unsupported_audio_fails_before_chat(self):
        b,t=self.make([])
        b.capabilities={'tools'}
        before=len(t.sent)
        with self.assertRaisesRegex(OllamaError,'native audio'):
            b.predict(audio=[0.0])
        self.assertEqual(len(t.sent),before)
        self.assertIsNone(b.pending)

    def test_overlong_and_invalid_audio_are_rejected(self):
        import numpy as np
        for data in [np.zeros(480001),np.zeros((2,2)),np.array([float('nan')])]:
            with self.subTest(shape=data.shape), self.assertRaises(ValueError):
                encode_audio(data)

    def test_connection_error_is_actionable(self):
        with patch('apm.ollama_backend.urlopen',side_effect=URLError('refused')):
            with self.assertRaisesRegex(OllamaError,'Open Ollama'):
                Transport().request('/api/show',{'model':'gemma4:e2b'})

    def test_missing_tool_support_fails_before_warmup(self):
        t=FakeTransport(capabilities=['completion'])
        with self.assertRaisesRegex(OllamaError,'tool calling'):
            OllamaBackend(transport=t).prepare()
        self.assertEqual(len(t.sent),1)

    def test_prepare_warms_configured_tools_and_audio_without_committing_a_turn(self):
        import soundfile as sf
        from apm.home import HomeController
        t = FakeTransport()
        b = OllamaBackend(transport=t)
        context = HomeController().context()
        context["clock"] = {"utc":"2026-10-07T12:00:00+00:00","timezone":"UTC"}
        b.configure_home(context)
        b.prepare(audio=True)
        path, payload = t.sent[-1]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(payload["tools"], context["tools"])
        self.assertEqual(payload["messages"][0]["content"], b.system)
        self.assertIn("12:00:00", payload["messages"][1]["content"])
        samples, rate = sf.read(io.BytesIO(base64.b64decode(payload["messages"][-1]["images"][0])))
        self.assertEqual((len(samples), rate), (16000, 16000))
        self.assertTrue((samples == 0).all())
        self.assertEqual(payload["options"]["num_predict"], 1)
        self.assertFalse(payload["stream"])
        self.assertEqual(b.turns, [])
        self.assertIsNone(b.pending)

    def test_prepare_audio_capability_failure_does_not_submit_silence(self):
        t = FakeTransport(capabilities=["completion", "tools"])
        with self.assertRaisesRegex(OllamaError, "native audio"):
            OllamaBackend(transport=t).prepare(audio=True)
        self.assertEqual([path for path, _ in t.sent], ["/api/show"])

    def test_blank_text_is_not_replaced_with_audio_instruction(self):
        b,t=self.make([])
        before=len(t.sent)
        for text in [None, "", "   "]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "non-empty"):
                b.predict(text=text)
        self.assertEqual(len(t.sent),before)
