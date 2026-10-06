import unittest
from apm.model import Gemma, ReplyStreamer, visible_reply

class Value:
    def __init__(self, ids): self.ids = ids
    def reshape(self, *_): return self
    def tolist(self): return self.ids

class Processor:
    def decode(self, ids, **kwargs): return ''.join(chr(i) for i in ids)

class ChatTests(unittest.TestCase):
    def test_prose_streams_but_tool_syntax_does_not(self):
        chunks = []
        stream = ReplyStreamer(Processor(), chunks.append)
        stream.put(Value([99]))  # prompt is not output
        for char in 'Hello!<|tool_call>call:set_lights{on:false}<tool_call|>':
            stream.put(Value([ord(char)]))
        self.assertEqual(''.join(chunks), 'Hello!')

    def test_history_keeps_complete_bounded_exchanges(self):
        model = Gemma.__new__(Gemma)
        model.history = []
        for i in range(6):
            model.pending = {'role': 'user', 'content': str(i)}
            model.commit({'content': 'reply'}, [])
        self.assertEqual(len(model.history), 8)
        self.assertEqual(model.history[0]['content'], '2')
        model.reset()
        self.assertEqual(model.history, [])

    def test_tool_result_is_in_history(self):
        model = Gemma.__new__(Gemma)
        model.history = []
        model.pending = {'role': 'user', 'content': 'close garage'}
        result = {'tool': 'set_garage', 'state': 'closed', 'simulated': True}
        model.commit({'tool_calls': [{'function': {'name': 'set_garage', 'arguments': {'target': 'closed'}}}]}, [result])
        self.assertEqual(model.history[-1]['tool_responses'][0]['response'], result)
