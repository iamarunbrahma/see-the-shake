"""API tests: offline by default (recorded responses replayed from fixtures/).

Tests marked `live` call the Meta Model API and cost money. They only run when selected: pytest -m live
"""
import pytest

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def pytest_configure(config):
    config.addinivalue_line("markers", "live: calls the Meta Model API (costs money); run with: pytest -m live")


def pytest_collection_modifyitems(config, items):
    if "live" in (config.getoption("markexpr") or ""):
        return
    skip = pytest.mark.skip(reason="live API test; run with: pytest -m live")
    for item in items:
        # the marker itself, not the keywords: a parametrize id such as [live] is a keyword too, and matching it
        # used to skip an offline test
        if item.get_closest_marker("live") is not None:
            item.add_marker(skip)
