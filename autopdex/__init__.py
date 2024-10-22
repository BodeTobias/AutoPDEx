# autopdex/__init__.py
import pytest


def run_tests():
    pytest.main(["-v", "tests/"])
