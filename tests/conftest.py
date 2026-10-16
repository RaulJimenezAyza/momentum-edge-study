"""Lets pytest run the check()-style tests.

Each test records failed checks in its module's FAILURES list instead of
raising, so the script runner can print every check. Under pytest that would
pass silently; this turns any check that failed during a test into a failure
of that test, with the names of the checks that failed.
"""

from __future__ import annotations

import pytest


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    failures = getattr(item.module, "FAILURES", None)
    before = len(failures) if failures is not None else 0
    result = yield
    if failures is not None and len(failures) > before:
        pytest.fail("failed checks: " + "; ".join(failures[before:]), pytrace=False)
    return result
