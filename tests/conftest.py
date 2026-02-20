"""Shared pytest configuration — stub out heavy ML imports globally."""
import sys
import types
from unittest.mock import MagicMock


def _make_insightface_stub():
    insightface = types.ModuleType("insightface")
    insightface_app = types.ModuleType("insightface.app")
    insightface_app_common = types.ModuleType("insightface.app.common")
    insightface_app_common.Face = object  # lightweight substitute for the type alias
    insightface.app = insightface_app
    insightface_app.common = insightface_app_common
    return insightface, insightface_app, insightface_app_common


def _stub_ml_packages():
    # insightface submodule hierarchy
    if "insightface" not in sys.modules:
        ins, ins_app, ins_app_common = _make_insightface_stub()
        sys.modules["insightface"] = ins
        sys.modules["insightface.app"] = ins_app
        sys.modules["insightface.app.common"] = ins_app_common

    # Stub modules.cluster_analysis so face_analyser doesn't pull in sklearn
    if "modules.cluster_analysis" not in sys.modules:
        cluster_stub = types.ModuleType("modules.cluster_analysis")
        cluster_stub.find_cluster_centroids = MagicMock(return_value=[])
        cluster_stub.find_closest_centroid = MagicMock(return_value=(0, None))
        sys.modules["modules.cluster_analysis"] = cluster_stub

    # tkinter needs TkVersion as a float
    if "tkinter" not in sys.modules:
        tk_mock = MagicMock()
        tk_mock.TkVersion = 8.6
        sys.modules["tkinter"] = tk_mock
    sys.modules.setdefault("_tkinter", MagicMock())

    for name in [
        "onnxruntime", "torch", "tensorflow",
        "gfpgan", "basicsr", "facexlib",
        "customtkinter",
    ]:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()


_stub_ml_packages()
