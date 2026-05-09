import unittest

from modules.camera_detection import NO_CAMERAS_FOUND, get_windows_cameras


class WindowsCameraDetectionTests(unittest.TestCase):
    def test_uses_directshow_devices_without_extra_probing(self):
        def filter_graph_factory():
            return _FilterGraph(["Integrated Camera", "USB Camera"])

        self.assertEqual(
            get_windows_cameras(filter_graph_factory),
            ([0, 1], ["Integrated Camera", "USB Camera"]),
        )

    def test_empty_directshow_list_returns_disabled_state(self):
        def filter_graph_factory():
            return _FilterGraph([])

        self.assertEqual(get_windows_cameras(filter_graph_factory), ([], [NO_CAMERAS_FOUND]))


class _FilterGraph:
    def __init__(self, devices):
        self._devices = devices

    def get_input_devices(self):
        return self._devices


if __name__ == "__main__":
    unittest.main()
