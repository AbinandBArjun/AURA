from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import test_qdrant_service as integration


class InMemoryCollections:
    def __init__(self, collections=None):
        self.collections = collections if collections is not None else {}
        self.lock = Lock()

    def service(self):
        client = Mock()
        client.create_collection.side_effect = self.create_collection
        client.delete_collection.side_effect = self.delete_collection
        return SimpleNamespace(VECTOR_SIZE=384, client=client, ensure_collection=Mock())

    def create_collection(self, *, collection_name, vectors_config):
        with self.lock:
            if collection_name in self.collections:
                raise RuntimeError("Collection already exists")
            self.collections[collection_name] = []
        return True

    def delete_collection(self, collection_name):
        with self.lock:
            del self.collections[collection_name]


def test_qdrant_keeps_preexisting_collection_and_points():
    sentinel = [{"id": 42, "payload": {"source": "existing"}}]
    store = InMemoryCollections({"aura_test_chunks": sentinel})
    service = store.service()

    with integration.isolated_collection(service) as active:
        assert active.COLLECTION_NAME != "aura_test_chunks"
        store.collections[active.COLLECTION_NAME].append({"id": 101})

    assert store.collections == {"aura_test_chunks": [{"id": 42, "payload": {"source": "existing"}}]}
    service.client.delete_collection.assert_called_once_with(service.COLLECTION_NAME)
    service.client.close.assert_called_once()


def test_qdrant_collision_does_not_claim_existing_collection(monkeypatch):
    name = "aura_test_chunks_collision"
    store = InMemoryCollections({name: [{"id": 42}]})
    service = store.service()
    monkeypatch.setattr(integration, "uuid4", lambda: SimpleNamespace(hex="collision"))

    with pytest.raises(RuntimeError, match="Collection already exists"):
        with integration.isolated_collection(service):
            pytest.fail("The fixture must not yield an existing collection")

    assert store.collections == {name: [{"id": 42}]}
    service.ensure_collection.assert_not_called()
    service.client.delete_collection.assert_not_called()
    service.client.close.assert_called_once()


def test_qdrant_cleanup_after_test_failure():
    store = InMemoryCollections()
    service = store.service()

    with pytest.raises(AssertionError, match="test failed"):
        with integration.isolated_collection(service):
            assert service.COLLECTION_NAME in store.collections
            raise AssertionError("test failed")

    assert not store.collections
    service.client.close.assert_called_once()


def test_qdrant_cleanup_after_partial_initialization():
    store = InMemoryCollections()
    service = store.service()
    service.ensure_collection.side_effect = RuntimeError("index setup failed")

    with pytest.raises(RuntimeError, match="index setup failed"):
        with integration.isolated_collection(service):
            pytest.fail("Failed initialization must not yield a collection")

    assert not store.collections
    service.client.delete_collection.assert_called_once_with(service.COLLECTION_NAME)
    service.client.close.assert_called_once()


def test_concurrent_qdrant_fixtures_use_separate_collections():
    store = InMemoryCollections()
    rendezvous = Barrier(2, timeout=5)

    def use_collection(point_id):
        with integration.isolated_collection(store.service()) as service:
            name = service.COLLECTION_NAME
            with store.lock:
                store.collections[name].append(point_id)
            rendezvous.wait()
            with store.lock:
                assert len(store.collections) == 2
                assert store.collections[name] == [point_id]
            rendezvous.wait()
            return name

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.map(use_collection, (1, 2))

    assert first != second
    assert not store.collections
