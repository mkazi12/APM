import json
import unittest

from apm.conversation import AssistantReply, reply_needs_answer


class ConversationTests(unittest.TestCase):
    def test_direct_questions_do_not_require_question_punctuation(self):
        for text in (
            "How can I help you today?", "how can i help for you today",
            "Which one exactly", "What time zone are you in?", "What time should I use",
            "Would you like the live recording?", "Do you mean the kitchen lights",
            "Is that correct?", "Can you repeat that", "Who is the reminder for?",
            "I found two matches. Which one would you like?", "Sure, which room?",
            "Kitchen or bedroom?", "Tomorrow morning?", "Anything else?",
            "What's the name of the song you would like me to play?",
            "What’s the time you want me to set the reminder for",
        ):
            with self.subTest(text=text):
                self.assertTrue(reply_needs_answer(text))

    def test_explicit_requests_for_answer(self):
        for text in (
            "Please specify the artist or version.", "Tell me which device you mean.",
            "Please choose one of those recordings.", "I need more details to find the device.",
            "Please let me know which time you prefer.", "Go ahead, I'm listening.",
            "Okay! Please repeat the song and artist.",
            "Please provide the name of the device.",
        ):
            with self.subTest(text=text):
                self.assertTrue(reply_needs_answer(text, []))

    def test_completion_or_unrelated_prose_does_not_open_followup(self):
        for text in (
            "", "   ", "Done.", "Hello!", "The lights are on.", "Your timer is set.",
            "Playing I'm On Fire by Bruce Springsteen.", "What a beautiful day!",
            "How lovely!", "I can help with timers.", "I don't know which room you mean.",
            "Please connect the Apple Music player.", "There is no matching song.",
            "The answer explains how to reset a timer.", "Let me know if you need anything else.",
            "When the timer ends, I'll let you know.", "What I can do is set timers.",
        ):
            with self.subTest(text=text):
                self.assertFalse(reply_needs_answer(text))

    def test_quoted_or_reported_questions_are_not_new_questions(self):
        for text in (
            'Playing "Who Are You?" by The Who.',
            "Found ‘Don't You Want Me?’ by The Human League.",
            "Playing 'Don't You Want Me?' by The Human League.",
            'The reminder says "What time is dinner?".',
            "You asked: Which room?", "Now playing: Who Are You?",
            "Playing Who Are You? by The Who.",
            "Use `what time?` as an example.", 'Example:\n```\nWhich room?\n```',
        ):
            with self.subTest(text=text):
                self.assertFalse(reply_needs_answer(text))
        self.assertTrue(reply_needs_answer('Should I play "Who Are You?"?'))

    def test_only_known_successful_ambiguity_result_requests_choice(self):
        ambiguity = {"tool":"play_music", "ok":True,
                     "data":{"status":"ambiguous", "candidates":[]}}
        self.assertTrue(reply_needs_answer("Please specify the artist or version.", [ambiguity]))
        for status in ("playing", "accepted", "matched", "not_found", "unknown", "failed"):
            result = {**ambiguity, "data":{"status":status, "track":{"title":"Which One?"}}}
            with self.subTest(status=status):
                self.assertFalse(reply_needs_answer("Which One?", [result]))
        self.assertFalse(reply_needs_answer("Which one?", [{**ambiguity, "ok":False}]))
        self.assertFalse(reply_needs_answer("Which one?", [{**ambiguity, "tool":"unknown"}]))
        self.assertFalse(reply_needs_answer("Which one?", [{"tool":"set_lights", "ok":True,
                                                             "label":"Which one?", "state":"on"}]))

    def test_reply_is_a_normal_string_with_explicit_metadata(self):
        reply = AssistantReply("Which room?", expects_reply=True)
        self.assertIsInstance(reply, str)
        self.assertEqual(reply, "Which room?")
        self.assertEqual(json.dumps(reply), '"Which room?"')
        self.assertTrue(reply.expects_reply)
        self.assertFalse(AssistantReply("Done.").expects_reply)
        with self.assertRaises(TypeError):
            AssistantReply("Which room?", expects_reply="yes")


if __name__ == "__main__":
    unittest.main()
