import importlib
import sys
import types
import unittest
from unittest.mock import patch


class FakeTclError(Exception):
    pass


class FakeWidget:
    def __init__(self):
        self.bindings = {}
        self.after_cancel_calls = []

    def bind(self, sequence, callback, add=None):
        self.bindings[sequence] = (callback, add)

    def after_cancel(self, after_id):
        self.after_cancel_calls.append(after_id)
        raise FakeTclError("widget has been destroyed")


class FakeTooltipWindow:
    def destroy(self):
        raise FakeTclError("tooltip has been destroyed")


def _load_tooltip_module():
    cv2 = types.ModuleType("cv2")
    cv2.IMREAD_COLOR = 1
    numpy = types.ModuleType("numpy")
    customtkinter = types.ModuleType("customtkinter")
    customtkinter.CTkBaseClass = object
    tkinter = types.ModuleType("tkinter")
    tkinter.TclError = FakeTclError
    with patch.dict(
        sys.modules,
        {
            "cv2": cv2,
            "numpy": numpy,
            "customtkinter": customtkinter,
            "tkinter": tkinter,
        },
    ):
        sys.modules.pop("modules.ui_tooltip", None)
        return importlib.import_module("modules.ui_tooltip")


class ToolTipDestroyTests(unittest.TestCase):
    def test_destroy_binding_cleans_pending_callback_and_window(self):
        tooltip_module = _load_tooltip_module()
        widget = FakeWidget()
        tooltip = tooltip_module.ToolTip(widget, "text")
        tooltip._after_id = "after-1"
        tooltip._tooltip_window = FakeTooltipWindow()

        self.assertIn("<Destroy>", widget.bindings)
        tooltip._on_destroy()

        self.assertEqual(widget.after_cancel_calls, ["after-1"])
        self.assertIsNone(tooltip._after_id)
        self.assertIsNone(tooltip._tooltip_window)


if __name__ == "__main__":
    unittest.main()
