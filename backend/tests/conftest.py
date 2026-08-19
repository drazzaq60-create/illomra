"""Shared test setup for the StudyMind backend suite.

- Puts the backend dir on sys.path so `import api` / `import rag_engine` work
  regardless of how pytest is invoked.
- Clears the per-IP rate limiter between tests so one test's requests can
  never push a later test over the 30 req/min window.
"""
import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import pytest


@pytest.fixture(autouse=True)
def _fresh_rate_limiter():
    import api
    api._rate_hits.clear()
    yield
    api._rate_hits.clear()
