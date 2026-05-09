import importlib
import sys
import types
import unittest
from unittest.mock import Mock, patch


class StubModule(types.ModuleType):
    def __getattr__(self, name):
        value = object
        setattr(self, name, value)
        return value


def _module(**attrs):
    mod = StubModule("stub")
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _install_import_stubs():
    ctk = _module(
        CTk=object,
        CTkImage=object,
        filedialog=_module(askopenfilename=lambda **_kwargs: ""),
    )
    ctk.__path__ = []
    sys.modules.setdefault("customtkinter", ctk)
    dropdown_cls = type(
        "DropdownMenu", (), {"_add_menu_commands": lambda self, *args, **kwargs: None}
    )
    sys.modules.setdefault("customtkinter.windows", _module())
    sys.modules.setdefault("customtkinter.windows.widgets", _module())
    sys.modules.setdefault(
        "customtkinter.windows.widgets.core_widget_classes",
        _module(DropdownMenu=dropdown_cls),
    )

    image_mod = _module(open=lambda *_args, **_kwargs: object(), fromarray=lambda frame: frame)
    image_ops_mod = _module(
        fit=lambda image, *_args, **_kwargs: image,
        contain=lambda image, *_args, **_kwargs: image,
    )
    sys.modules.setdefault("PIL", _module(Image=image_mod, ImageOps=image_ops_mod))
    sys.modules.setdefault("PIL.Image", image_mod)
    sys.modules.setdefault("PIL.ImageOps", image_ops_mod)

    sys.modules.setdefault(
        "cv2",
        _module(
            IMREAD_COLOR=1,
            imread=lambda *_args, **_kwargs: None,
            imdecode=lambda *_args, **_kwargs: None,
            imencode=lambda *_args, **_kwargs: (
                True,
                _module(tofile=lambda *_args, **_kwargs: None),
            ),
            VideoCapture=lambda *_args, **_kwargs: _module(
                isOpened=lambda: False, release=lambda: None
            ),
            destroyAllWindows=lambda: None,
            COLOR_BGR2RGB=1,
            CAP_PROP_POS_FRAMES=1,
            FONT_HERSHEY_SIMPLEX=0,
            putText=lambda *args, **kwargs: None,
        ),
    )
    sys.modules.setdefault("numpy", _module(uint8=object, fromfile=lambda *_args, **_kwargs: b""))
    sys.modules.setdefault("requests", _module(get=lambda *args, **kwargs: None))

    globals_mod = _module(
        file_types=(("Images", "*.png"), ("Videos", "*.mp4")),
        source_path="/missing/source.png",
        target_path="/tmp/target.png",
        nsfw_filter=False,
        frame_processors=[],
        map_faces=False,
        live_mirror=False,
        many_faces=False,
        fp_ui={},
        show_fps=False,
    )
    sys.modules["modules.globals"] = globals_mod
    import modules
    modules.globals = globals_mod
    sys.modules["modules.metadata"] = _module()
    modules.metadata = sys.modules["modules.metadata"]
    sys.modules["modules.gpu_processing"] = _module(
        gpu_cvt_color=lambda frame, *_args, **_kwargs: frame,
        gpu_resize=lambda frame, *args, **kwargs: frame,
        gpu_flip=lambda frame, *_args, **_kwargs: frame,
    )
    sys.modules["modules.face_analyser"] = _module(
        get_one_face=Mock(name="get_one_face"),
        get_many_faces=lambda *_args, **_kwargs: [],
        detect_one_face_fast=lambda *_args, **_kwargs: None,
        detect_many_faces_fast=lambda *_args, **_kwargs: [],
        get_unique_faces_from_target_image=lambda *_args, **_kwargs: [],
        get_unique_faces_from_target_video=lambda *_args, **_kwargs: [],
        add_blank_map=lambda *_args, **_kwargs: None,
        has_valid_map=lambda *_args, **_kwargs: False,
        simplify_maps=lambda *_args, **_kwargs: None,
    )
    sys.modules["modules.capturer"] = _module(
        get_video_frame=lambda *_args, **_kwargs: None,
        get_video_frame_total=lambda *_args, **_kwargs: 0,
    )
    sys.modules["modules.processors.frame.core"] = _module(
        get_frame_processors_modules=lambda *_args, **_kwargs: []
    )
    sys.modules["modules.utilities"] = _module(
        is_image=lambda path: True,
        is_video=lambda path: False,
        resolve_relative_path=lambda path: path,
        has_image_extension=lambda path: True,
    )
    sys.modules["modules.video_capture"] = _module(VideoCapturer=object)
    sys.modules["modules.gettext"] = _module(LanguageManager=lambda lang: _module(_=lambda text: text))
    sys.modules["modules.ui_tooltip"] = _module(ToolTip=object)


def _load_ui():
    _install_import_stubs()
    sys.modules.pop("modules.ui", None)
    return importlib.import_module("modules.ui")


class SourceFaceGuardTests(unittest.TestCase):
    def test_unreadable_source_image_does_not_call_face_analyser(self):
        ui = _load_ui()
        ui.update_status = Mock()
        ui.modules.globals.source_path = "/missing/source.png"

        with patch.object(ui.cv2, "imread", return_value=None):
            self.assertIsNone(ui.get_source_face())

        ui.get_one_face.assert_not_called()
        ui.update_status.assert_called_with("Source image could not be read.")


if __name__ == "__main__":
    unittest.main()
