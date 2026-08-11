"""
Reply language must follow the question, not the sources.

Reported from real use: an English question against an Arabic corpus came back in
Arabic. A general "match the user's language" rule sat at position one of a long
prompt and lost to eighty-eight chunks of immediate Arabic context, so the rule is
now concrete — the script is named, the question is quoted back, and the directive
is repeated last, where the model weights it most.
"""

import unittest

from app.utils.helpers import dominant_script
from app.utils.prompts import get_system_prompt

ARABIC = "ما هي شروط التسجيل كباحث عن عمل؟"
ENGLISH = "What are the eligibility conditions to register as a job seeker?"


class ScriptDetectionTest(unittest.TestCase):
    def test_english(self):
        self.assertEqual(dominant_script(ENGLISH), "Latin")

    def test_arabic(self):
        self.assertEqual(dominant_script(ARABIC), "Arabic")

    def test_cyrillic(self):
        self.assertEqual(dominant_script("Каковы условия регистрации?"), "Cyrillic")

    def test_cjk(self):
        self.assertEqual(dominant_script("登録条件は何ですか"), "CJK")

    def test_mixed_falls_to_the_majority_script(self):
        """A stray Latin token in an Arabic question must not flip the reply."""
        self.assertEqual(dominant_script(ARABIC + " (PDF)"), "Arabic")

    def test_latin_with_a_stray_arabic_word_stays_latin(self):
        self.assertEqual(dominant_script("What does وزارة mean in this report?"), "Latin")

    def test_empty_defaults_to_latin(self):
        self.assertEqual(dominant_script(""), "Latin")

    def test_digits_and_symbols_only(self):
        self.assertEqual(dominant_script("2 + 2 = ?"), "Latin")


class PromptLanguageDirectiveTest(unittest.TestCase):
    def prompt(self, query, context=""):
        return get_system_prompt(context, "No tables.", [], "", query)

    def test_latin_is_never_named_as_a_language(self):
        """
        Naming it produced replies in actual Latin: "Informationes de condicionibus
        eligibilitatis...". The question itself is the anchor instead.
        """
        self.assertNotIn("Latin", self.prompt(ENGLISH))

    def test_a_non_latin_script_is_named_because_it_is_unambiguous(self):
        self.assertIn("Arabic script", self.prompt(ARABIC))

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
