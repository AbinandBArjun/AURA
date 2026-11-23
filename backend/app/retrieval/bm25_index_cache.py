from collections import OrderedDict
from threading import RLock
from typing import Any, Callable


class BM25IndexCache:
    """Thread-safe, bounded LRU cache for workspace BM25 indexes."""

    def __init__(self, max_entries: int = 32):
        if max_entries <= 0:
            raise ValueError("Cache capacity must be positive.")

        self.max_entries = max_entries
        self._entries: OrderedDict[int, Any] = OrderedDict()
        self._lock = RLock()

    def get_or_create(
        self,
        workspace_id: int,
        factory: Callable[[], Any],
    ) -> Any:
        with self._lock:
            if workspace_id in self._entries:
                self._entries.move_to_end(workspace_id)
                return self._entries[workspace_id]

            entry = factory()

            self._entries[workspace_id] = entry
            self._entries.move_to_end(workspace_id)

            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

            return entry

    def invalidate(self, workspace_id: int) -> None:
        with self._lock:
            self._entries.pop(workspace_id, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


bm25_index_cache = BM25IndexCache()
