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
    def test_rate_limit_is_transient(self):
        self.assertTrue(_is_transient(Status429("rate limit exceeded")))

    def test_overload_is_transient(self):
        self.assertTrue(_is_transient(Status503("model is overloaded")))

    def test_gemini_quota_wording_is_transient(self):
        self.assertTrue(_is_transient(Exception("429 RESOURCE_EXHAUSTED, try again later")))

    def test_gemini_high_demand_is_transient(self):
        self.assertTrue(_is_transient(Exception("503 UNAVAILABLE: high demand")))

    def test_out_of_credit_is_not_transient(self):
        """A funding failure is also a 429, but retrying it only wastes calls."""
        self.assertFalse(
            _is_transient(Status429("insufficient_quota: You have no credits remaining"))
        )

    def test_billing_wording_is_not_transient(self):
        self.assertFalse(_is_transient(Exception("credit_balance_exhausted")))

    def test_bad_request_is_not_transient(self):
        self.assertFalse(_is_transient(Exception("400 INVALID_ARGUMENT: bad schema")))


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

    async def test_funding_failure_is_not_retried(self):
        """The whole point: never burn extra billable calls on a dead account."""
        call = AsyncMock(side_effect=Status429("insufficient_quota, no credits remaining"))
        with self.assertRaises(Status429):
            await _with_retry(call, "test")
        self.assertEqual(call.await_count, 1)

    async def test_permanent_error_is_not_retried(self):
        call = AsyncMock(side_effect=ValueError("400 INVALID_ARGUMENT"))
        with self.assertRaises(ValueError):
            await _with_retry(call, "test")
        self.assertEqual(call.await_count, 1)


if __name__ == "__main__":
    unittest.main()
