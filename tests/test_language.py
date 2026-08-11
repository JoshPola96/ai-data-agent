"""
Reply language must follow the question, not the sources.

Reported from real use: an English question against an Arabic corpus came back in
Arabic. A general "match the user's language" rule sat at position one of a long
prompt and lost to eighty-eight chunks of immediate Arabic context, so the rule is
now concrete — the script is named, the question is quoted back, and the directive
is repeated last, where the model weights it most.
"""

import unittest

from app.utils.prompts import get_system_prompt

ARABIC = "ما هي شروط التسجيل كباحث عن عمل؟"
ENGLISH = "What are the eligibility conditions to register as a job seeker?"


class PromptLanguageDirectiveTest(unittest.TestCase):
    def prompt(self, query, context=""):
        return get_system_prompt(context, "No tables.", [], "", query)

    def test_no_language_or_script_is_ever_named(self):
        """
        A hand-rolled detector naming the script produced replies in actual Latin:
        "Informationes de condicionibus eligibilitatis...". The quoted question is
        already in the target language, so nothing needs classifying.
        """
        prompt = self.prompt(ENGLISH)
        for word in ("Latin", "Arabic script", "Cyrillic", "CJK"):
            self.assertNotIn(word, prompt)

    def test_an_arabic_question_is_quoted_in_arabic(self):
        self.assertIn(ARABIC, self.prompt(ARABIC))

    def test_the_question_is_quoted_back(self):
        self.assertIn("eligibility conditions", self.prompt(ENGLISH))

    def test_the_directive_appears_twice(self):
        """Once up front, once last: the final instruction is the one that sticks."""
        self.assertEqual(self.prompt(ENGLISH).count("Reply in the same language"), 2)

    def test_the_directive_survives_a_large_foreign_context(self):
        arabic_context = ARABIC * 400
        prompt = self.prompt(ENGLISH, context=arabic_context)
        self.assertIn("eligibility conditions", prompt)
        self.assertGreater(
            prompt.rindex("Reply in the same language"),
            prompt.rindex(ARABIC),
            "the directive must come after the foreign context, not before it",
        )

    def test_it_states_that_sources_do_not_set_the_language(self):
        self.assertIn("never adopt their language", self.prompt(ENGLISH))

    def test_a_multiline_question_is_flattened_when_quoted(self):
        prompt = self.prompt("First line\nsecond line")
        self.assertIn("First line second line", prompt)

    def test_no_query_still_produces_a_prompt(self):
        self.assertIn("Reply in the same language", get_system_prompt("", "No tables.", [], ""))


if __name__ == "__main__":
    unittest.main()
