"""Tests for modules/model_loader.py."""

import threading
import pytest
from modules.model_loader import ModelHolder


def test_get_returns_loaded_model():
    holder = ModelHolder()
    result = holder.get(loader_fn=lambda: "model")
    assert result == "model"


def test_get_caches_model():
    call_count = 0

    def loader():
        nonlocal call_count
        call_count += 1
        return "model"

    holder = ModelHolder()
    holder.get(loader_fn=loader)
    holder.get(loader_fn=loader)
    assert call_count == 1


def test_warmup_called_once():
    warmup_calls = []

    def warmup(model):
        warmup_calls.append(model)

    holder = ModelHolder()
    holder.get(loader_fn=lambda: "m", warmup_fn=warmup)
    holder.get(loader_fn=lambda: "m", warmup_fn=warmup)
    assert warmup_calls == ["m"]


def test_loader_error_propagates_and_retries():
    attempt = 0

    def loader():
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            raise ValueError("fail")
        return "ok"

    holder = ModelHolder()
    with pytest.raises(ValueError):
        holder.get(loader_fn=loader)

    assert not holder.is_loaded
    result = holder.get(loader_fn=loader)
    assert result == "ok"
    assert holder.is_loaded


def test_clear_resets():
    holder = ModelHolder()
    holder.get(loader_fn=lambda: "a")
    assert holder.is_loaded
    holder.clear()
    assert not holder.is_loaded
    result = holder.get(loader_fn=lambda: "b")
    assert result == "b"


def test_thread_safety():
    """Multiple threads calling get() concurrently should only invoke loader once."""
    call_count = 0
    lock = threading.Lock()

    def loader():
        nonlocal call_count
        with lock:
            call_count += 1
        return "model"

    holder = ModelHolder()
    threads = [threading.Thread(target=lambda: holder.get(loader_fn=loader)) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert call_count == 1
