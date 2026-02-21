"""Reusable lazy-singleton model loader with double-check locking and optional warmup."""

import threading
from typing import Any, Callable, Optional


class ModelHolder:
    """Thread-safe lazy singleton for heavy model objects.

    Usage::

        _model = ModelHolder()

        def get_face_enhancer():
            return _model.get(loader_fn=_load_gfpgan, warmup_fn=_warmup_gfpgan)
    """

    def __init__(self) -> None:
        self._instance: Any = None
        self._lock = threading.Lock()

    def get(
        self,
        loader_fn: Callable[[], Any],
        warmup_fn: Optional[Callable[[Any], None]] = None,
    ) -> Any:
        """Return the cached model, loading it on first access.

        Implements double-check locking: fast path avoids the lock when the
        model is already loaded; slow path holds the lock while loading.

        Raises whatever ``loader_fn`` raises on failure (the holder remains
        empty so a subsequent call will retry).
        """
        if self._instance is not None:
            return self._instance

        with self._lock:
            if self._instance is not None:
                return self._instance

            model = loader_fn()
            if warmup_fn is not None:
                warmup_fn(model)
            self._instance = model

        return self._instance

    def clear(self) -> None:
        """Reset the holder so the next ``get()`` reloads the model."""
        with self._lock:
            self._instance = None

    @property
    def is_loaded(self) -> bool:
        return self._instance is not None
