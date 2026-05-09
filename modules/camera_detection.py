from collections.abc import Callable
from typing import Protocol


NO_CAMERAS_FOUND = "No cameras found"


class _FilterGraph(Protocol):
    def get_input_devices(self) -> list[str]:
        ...


def get_windows_cameras(filter_graph_factory: Callable[[], _FilterGraph]) -> tuple[list[int], list[str]]:
    """Return DirectShow camera indices and names using a zero-argument FilterGraph factory."""
    devices = filter_graph_factory().get_input_devices()
    if not devices:
        return [], [NO_CAMERAS_FOUND]

    return list(range(len(devices))), devices
