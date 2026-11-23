from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pytest

from app.retrieval.bm25_index_cache import BM25IndexCache


@pytest.fixture
def cache():
    return BM25IndexCache(max_entries=2)


def test_cache_builds_entry_once(cache):
    calls = 0

    def factory():
        nonlocal calls
        calls += 1
        return {"index": "workspace-1"}

    first = cache.get_or_create(1, factory)
    second = cache.get_or_create(1, factory)

    assert first is second
    assert calls == 1


def test_cache_isolates_workspaces(cache):
    first = cache.get_or_create(1, lambda: "workspace-1")
    second = cache.get_or_create(2, lambda: "workspace-2")

    assert first == "workspace-1"
    assert second == "workspace-2"
    assert len(cache) == 2


def test_invalidate_removes_only_requested_workspace(cache):
    cache.get_or_create(1, lambda: "first")
    cache.get_or_create(2, lambda: "second")

    cache.invalidate(1)

    assert len(cache) == 1
    assert cache.get_or_create(2, lambda: "unexpected") == "second"
    assert cache.get_or_create(1, lambda: "replacement") == "replacement"


def test_cache_evicts_least_recently_used_entry(cache):
    cache.get_or_create(1, lambda: "first")
    cache.get_or_create(2, lambda: "second")

    cache.get_or_create(1, lambda: "unused")
    cache.get_or_create(3, lambda: "third")

    assert len(cache) == 2
    assert cache.get_or_create(1, lambda: "unexpected") == "first"
    assert cache.get_or_create(2, lambda: "replacement") == "replacement"


def test_clear_removes_all_entries(cache):
    cache.get_or_create(1, lambda: "first")
    cache.get_or_create(2, lambda: "second")

    cache.clear()

    assert len(cache) == 0


def test_invalid_capacity_raises_error():
    with pytest.raises(ValueError, match="Cache capacity"):
        BM25IndexCache(max_entries=0)


def test_concurrent_requests_build_entry_once(cache):
    calls = 0
    calls_lock = Lock()

    def factory():
        nonlocal calls

        with calls_lock:
            calls += 1

        return {"index": "shared"}

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: cache.get_or_create(1, factory),
                range(20),
            )
        )

    assert calls == 1
    assert all(result is results[0] for result in results)
