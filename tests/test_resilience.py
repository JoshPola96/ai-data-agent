"""Transient-fault retry: what gets retried, what fails through immediately."""

import unittest
from unittest.mock import AsyncMock, patch

import app.services.llm as llm_mod
from app.services.llm import _is_transient, _with_retry


class Status429(Exception):
    status_code = 429


class Status503(Exception):
    status_code = 503


class ClassifyTest(unittest.TestCase):
    """
    Retry is decided by HTTP status alone.

    An earlier version matched provider message text and misread Gemini's recoverable
    rate limit as terminal, because that message mentions billing. Prose is not an
    interface; the status code is.
    """

    def test_rate_limit_is_retryable(self):
        self.assertTrue(_is_transient(Status429("rate limit exceeded")))

    def test_overload_is_retryable(self):
        self.assertTrue(_is_transient(Status503("model is overloaded")))

    def test_an_exhausted_account_is_retried_too(self):
        """
        It will not clear, but a 429 is refused before any completion is generated,
        so retrying costs latency rather than money.
        """
        self.assertTrue(
            _is_transient(Status429("insufficient_quota: no credits remaining"))
        )

    def test_a_bad_request_is_not_retryable(self):
        class Status400(Exception):
            status_code = 400

        self.assertFalse(_is_transient(Status400("INVALID_ARGUMENT: bad schema")))

    def test_google_style_code_attribute_is_read(self):
        class GoogleError(Exception):
            code = 503

        self.assertTrue(_is_transient(GoogleError("UNAVAILABLE")))

    def test_an_exception_without_a_status_is_not_retryable(self):
        self.assertFalse(_is_transient(ValueError("something local broke")))

    def test_message_text_is_never_consulted(self):
        """Alarming words in the message must not make a 400 retryable."""

        class Status400(Exception):
            status_code = 400

        self.assertFalse(_is_transient(Status400("overloaded, try again, timeout")))


class RetryTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        p = patch.multiple(llm_mod.settings, LLM_MAX_RETRIES=3, LLM_RETRY_BASE_DELAY=0.0)
        p.start()
        self.addCleanup(p.stop)

    async def test_succeeds_without_retrying(self):
        call = AsyncMock(return_value="ok")
        self.assertEqual(await _with_retry(call, "test"), "ok")
        self.assertEqual(call.await_count, 1)

    async def test_transient_fault_is_retried_then_succeeds(self):
        call = AsyncMock(side_effect=[Status503("overloaded"), "ok"])
        self.assertEqual(await _with_retry(call, "test"), "ok")
        self.assertEqual(call.await_count, 2)

    async def test_retries_are_bounded(self):
        call = AsyncMock(side_effect=Status503("overloaded"))
        with self.assertRaises(Status503):
            await _with_retry(call, "test")
        self.assertEqual(call.await_count, 3)

    async def test_an_error_without_a_status_fails_immediately(self):
        call = AsyncMock(side_effect=ValueError("400 INVALID_ARGUMENT"))
        with self.assertRaises(ValueError):
            await _with_retry(call, "test")
        self.assertEqual(call.await_count, 1)


if __name__ == "__main__":
    unittest.main()
